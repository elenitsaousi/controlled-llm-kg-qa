#!/usr/bin/env python3
"""Evaluate confidence-aware answer/clarification routing for KGQA results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ranking.query_contract import extract_question_contract


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Results JSON must be an object.")
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


def _ranked_candidates(
    detail: Dict[str, object],
    score_key: str,
    *,
    sort_by_score: bool,
) -> List[Dict[str, object]]:
    candidates = [c for c in list(detail.get("candidates") or []) if isinstance(c, dict)]
    if sort_by_score:
        return sorted(candidates, key=lambda c: _score(c, score_key), reverse=True)
    return candidates


def _case_rows(
    results: Dict[str, object],
    score_key: str,
    *,
    sort_by_score: bool,
    dataset_by_id: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for detail in results.get("details") or []:
        if not isinstance(detail, dict):
            continue
        qid = str(detail.get("id", "") or "")
        dataset_row = dataset_by_id.get(qid, {})
        candidates = _ranked_candidates(detail, score_key, sort_by_score=sort_by_score)
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
        question = _question(detail)
        try:
            contract = extract_question_contract(question).to_dict()
        except Exception:
            contract = {}
        rows.append(
            {
                "id": qid,
                "question": question,
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
                "question_contract": contract,
                "score1": score1,
                "score2": score2,
                "margin": score1 - score2,
                "top1_correct": _label(top1) == "correct",
                "any_correct": any(_label(candidate) == "correct" for candidate in candidates),
                "top3": top3,
            }
        )
    return rows


def _distribution(rows: List[Dict[str, object]], key: str, limit: int = 20) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for row in rows:
        value = row.get(key)
        if isinstance(value, (list, set, tuple)):
            for item in value:
                counter[str(item)] += 1
        else:
            counter[str(value or "unknown")] += 1
    return counter.most_common(limit)


def _contract_distribution(rows: List[Dict[str, object]], axis: str, limit: int = 20) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for row in rows:
        contract = dict(row.get("question_contract") or {})
        value = contract.get(axis)
        if isinstance(value, list):
            if value:
                for item in value:
                    counter[str(item)] += 1
            else:
                counter["none"] += 1
        elif value:
            counter[str(value)] += 1
        else:
            counter["none"] += 1
    return counter.most_common(limit)


def _policy_bucket_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    correct = sum(1 for row in rows if row["top1_correct"])
    any_correct = sum(1 for row in rows if row["any_correct"])
    return {
        "count": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "any_correct": any_correct,
        "any_correct_rate": (any_correct / total) if total else 0.0,
        "families": _distribution(rows, "family"),
        "ambiguity_labels": _distribution(rows, "ambiguity_label"),
        "metrics": _contract_distribution(rows, "metrics"),
        "aggregation": _contract_distribution(rows, "aggregation"),
        "scopes": _contract_distribution(rows, "scopes"),
        "dimensions": _contract_distribution(rows, "dimensions"),
        "answer_shape": _contract_distribution(rows, "answer_shape"),
    }


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
    dataset_path: str,
    score_key: str,
    sort_by_score: bool,
    policy_min_margin: float,
    policy_min_score: float,
    margin_thresholds: List[float],
    score_thresholds: List[float],
    low_confidence_limit: int,
    high_confidence_wrong_limit: int,
) -> Dict[str, object]:
    results = _load_json(results_path)
    dataset_by_id = _load_dataset(dataset_path)
    rows = _case_rows(
        results,
        score_key,
        sort_by_score=sort_by_score,
        dataset_by_id=dataset_by_id,
    )
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
        ("score_gt_0.95", 0.95, None),
        ("score_0.90_to_0.95", 0.90, 0.95),
        ("score_0.85_to_0.90", 0.85, 0.90),
        ("score_0.80_to_0.85", 0.80, 0.85),
        ("score_0.60_to_0.80", 0.60, 0.80),
        ("score_lte_0.60", None, 0.60),
    ]

    margin_rows: Dict[str, List[Dict[str, object]]] = {name: [] for name, _, _ in margin_buckets}
    score_rows: Dict[str, List[Dict[str, object]]] = {name: [] for name, _, _ in score_buckets}
    for row in rows:
        margin_rows[_bucket(float(row["margin"]), margin_buckets)].append(row)
        score_rows[_bucket(float(row["score1"]), score_buckets)].append(row)

    low_confidence = sorted(rows, key=lambda row: (float(row["margin"]), float(row["score1"])))[:low_confidence_limit]
    high_confidence_wrong = [
        row
        for row in sorted(rows, key=lambda row: (float(row["score1"]), float(row["margin"])), reverse=True)
        if not row["top1_correct"]
    ][:high_confidence_wrong_limit]
    auto_answer = [
        row
        for row in rows
        if float(row["margin"]) >= policy_min_margin and float(row["score1"]) >= policy_min_score
    ]
    clarification = [row for row in rows if row not in auto_answer]

    return {
        "inputs": {
            "results": results_path,
            "dataset": dataset_path,
            "score_key": score_key,
            "sort_by_score": sort_by_score,
            "policy_min_margin": policy_min_margin,
            "policy_min_score": policy_min_score,
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
        "policy_buckets": {
            "auto_answer": _policy_bucket_summary(auto_answer),
            "clarification": _policy_bucket_summary(clarification),
        },
        "low_confidence_examples": low_confidence,
        "high_confidence_wrong_examples": high_confidence_wrong,
    }


def _format_pct(value: object) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _best_policy_at_target(report: Dict[str, object], target_accuracy: float) -> Optional[Dict[str, object]]:
    eligible = [
        dict(row)
        for row in report.get("threshold_sweep") or []
        if float(row.get("accuracy_when_answered") or 0.0) >= target_accuracy
        and int(row.get("answered") or 0) > 0
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            float(row.get("coverage") or 0.0),
            float(row.get("accuracy_when_answered") or 0.0),
            int(row.get("answered") or 0),
        ),
        reverse=True,
    )[0]


def _append_distribution(lines: List[str], title: str, values: Iterable[Tuple[str, int]]) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append("| Value | Count |")
    lines.append("|---|---:|")
    for value, count in values:
        lines.append(f"| `{value}` | {count} |")
    lines.append("")


def write_markdown(report: Dict[str, object], out_path: str) -> None:
    summary = dict(report.get("summary") or {})
    lines: List[str] = []
    lines.append("# Confidence-Aware Routing Analysis")
    lines.append("")
    lines.append(f"- Results: `{dict(report.get('inputs') or {}).get('results')}`")
    lines.append(f"- Dataset: `{dict(report.get('inputs') or {}).get('dataset')}`")
    lines.append(f"- Score key: `{dict(report.get('inputs') or {}).get('score_key')}`")
    lines.append(f"- Sort by score: `{dict(report.get('inputs') or {}).get('sort_by_score')}`")
    lines.append(
        "- Policy: "
        f"margin >= {float(dict(report.get('inputs') or {}).get('policy_min_margin') or 0.0):.2f}, "
        f"score >= {float(dict(report.get('inputs') or {}).get('policy_min_score') or 0.0):.2f}"
    )
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
    lines.append("## Best Policies at Target Accuracy")
    lines.append("")
    lines.append("| Target accuracy | Min margin | Min score | Answered | Coverage | Accuracy | Clarified |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for target in (0.90, 0.85, 0.80):
        row = _best_policy_at_target(report, target)
        if not row:
            lines.append(f"| {target:.2f} | - | - | 0 | 0.000 | - | {summary.get('total', 0)} |")
            continue
        lines.append(
            f"| {target:.2f} | {row['min_margin']:.2f} | {row['min_score']:.2f} | "
            f"{row['answered']} | {_format_pct(row['coverage'])} | "
            f"{_format_pct(row['accuracy_when_answered'])} | {row['clarified']} |"
        )
    lines.append("")
    lines.append("## Policy Buckets")
    lines.append("")
    lines.append("| Bucket | Count | Correct | Accuracy | Any correct | Any-correct rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    policy_buckets = dict(report.get("policy_buckets") or {})
    for name in ("auto_answer", "clarification"):
        row = dict(policy_buckets.get(name) or {})
        lines.append(
            f"| `{name}` | {row.get('count', 0)} | {row.get('correct', 0)} | "
            f"{_format_pct(row.get('accuracy'))} | {row.get('any_correct', 0)} | "
            f"{_format_pct(row.get('any_correct_rate'))} |"
        )
    lines.append("")
    for name in ("auto_answer", "clarification"):
        row = dict(policy_buckets.get(name) or {})
        lines.append(f"## {name.replace('_', ' ').title()} Composition")
        lines.append("")
        _append_distribution(lines, "Families", row.get("families") or [])
        _append_distribution(lines, "Ambiguity Labels", row.get("ambiguity_labels") or [])
        _append_distribution(lines, "Metrics", row.get("metrics") or [])
        _append_distribution(lines, "Aggregation", row.get("aggregation") or [])
        _append_distribution(lines, "Scopes", row.get("scopes") or [])
        _append_distribution(lines, "Dimensions", row.get("dimensions") or [])
        _append_distribution(lines, "Answer Shape", row.get("answer_shape") or [])
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
    lines.append("## High-Confidence Wrong Examples")
    lines.append("")
    for row in report.get("high_confidence_wrong_examples") or []:
        lines.append(f"### {row.get('id')}")
        lines.append("")
        lines.append(f"Question: {row.get('question')}")
        lines.append("")
        lines.append(
            f"- score1={float(row.get('score1', 0.0)):.4f}, "
            f"score2={float(row.get('score2', 0.0)):.4f}, "
            f"margin={float(row.get('margin', 0.0)):.4f}"
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
    parser.add_argument("--dataset", default="")
    parser.add_argument("--score-key", default="ml_score", choices=["ml_score", "selection_score"])
    parser.add_argument(
        "--sort-by-score",
        action="store_true",
        help="Diagnostic mode: sort candidates by score before computing routing metrics.",
    )
    parser.add_argument("--policy-min-margin", type=float, default=0.0)
    parser.add_argument("--policy-min-score", type=float, default=0.90)
    parser.add_argument("--margin-thresholds", default="0.00,0.03,0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--score-thresholds", default="0.00,0.40,0.50,0.60,0.70,0.80,0.85,0.90,0.95")
    parser.add_argument("--low-confidence-limit", type=int, default=12)
    parser.add_argument("--high-confidence-wrong-limit", type=int, default=12)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(
        results_path=args.results,
        dataset_path=args.dataset,
        score_key=args.score_key,
        sort_by_score=args.sort_by_score,
        policy_min_margin=args.policy_min_margin,
        policy_min_score=args.policy_min_score,
        margin_thresholds=_parse_float_list(args.margin_thresholds),
        score_thresholds=_parse_float_list(args.score_thresholds),
        low_confidence_limit=args.low_confidence_limit,
        high_confidence_wrong_limit=args.high_confidence_wrong_limit,
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
    print("Best policies at target accuracy:")
    for target in (0.90, 0.85, 0.80):
        row = _best_policy_at_target(report, target)
        if not row:
            print(f"  target>={target:.2f}: no non-empty policy")
            continue
        print(
            f"  target>={target:.2f}: margin>={row['min_margin']:.2f}, "
            f"score>={row['min_score']:.2f}, answered={row['answered']} "
            f"coverage={row['coverage']:.3f}, accuracy={row['accuracy_when_answered']:.3f}"
        )
    print("Policy buckets:")
    for name, row in dict(report.get("policy_buckets") or {}).items():
        print(
            f"  {name}: {row['correct']}/{row['count']} "
            f"accuracy={row['accuracy']:.3f}, any={row['any_correct_rate']:.3f}"
        )
        print(f"    top families: {row['families'][:5]}")
        print(f"    top aggregations: {row['aggregation'][:5]}")
        print(f"    top scopes: {row['scopes'][:5]}")
        print(f"    top dimensions: {row['dimensions'][:8]}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
