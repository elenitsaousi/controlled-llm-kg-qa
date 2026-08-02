#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _dataset_by_id(path: str) -> Dict[str, Dict[str, object]]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Dataset must be a JSON list: {path}")
    return {str(row.get("id", "")).strip(): row for row in rows if isinstance(row, dict)}


def _details_by_id(path: str) -> Dict[str, Dict[str, object]]:
    payload = _load_json(path)
    return {
        str(detail.get("id", "")).strip(): detail
        for detail in list(payload.get("details") or [])
        if isinstance(detail, dict) and str(detail.get("id", "")).strip()
    }


def _family(qid: str, dataset: Dict[str, Dict[str, object]], detail: Dict[str, object]) -> str:
    row = dataset.get(qid, {})
    return (
        str(
            row.get("family")
            or row.get("topic")
            or row.get("family_id")
            or detail.get("family")
            or detail.get("topic")
            or "unknown"
        ).strip()
        or "unknown"
    )


def derive_policy(
    *,
    raw_results: str,
    selected_results: str,
    dataset_path: str,
    min_delta_questions: int,
    min_gain_loss_ratio: float,
) -> Dict[str, object]:
    dataset = _dataset_by_id(dataset_path)
    raw = _details_by_id(raw_results)
    selected = _details_by_id(selected_results)
    common_ids = sorted(set(raw) & set(selected))
    if not common_ids:
        raise RuntimeError("No overlapping question IDs between raw and selected results.")

    stats: Dict[str, Counter] = defaultdict(Counter)
    for qid in common_ids:
        raw_detail = raw[qid]
        selected_detail = selected[qid]
        fam = _family(qid, dataset, raw_detail)
        raw_correct = bool(raw_detail.get("top1_correct"))
        selected_correct = bool(selected_detail.get("top1_correct"))
        stats[fam]["total"] += 1
        stats[fam]["raw_correct"] += int(raw_correct)
        stats[fam]["selected_correct"] += int(selected_correct)
        if raw_correct and selected_correct:
            stats[fam]["both_correct"] += 1
        elif raw_correct and not selected_correct:
            stats[fam]["losses"] += 1
        elif not raw_correct and selected_correct:
            stats[fam]["gains"] += 1
        else:
            stats[fam]["both_wrong"] += 1

    family_rows: List[Dict[str, object]] = []
    allowed_families: List[str] = []
    for fam in sorted(stats):
        row = dict(stats[fam])
        total = int(row.get("total", 0))
        raw_correct = int(row.get("raw_correct", 0))
        selected_correct = int(row.get("selected_correct", 0))
        gains = int(row.get("gains", 0))
        losses = int(row.get("losses", 0))
        delta = selected_correct - raw_correct
        gain_loss_ratio = gains / losses if losses else (float("inf") if gains else 0.0)
        enabled = (
            delta >= min_delta_questions
            and gains > 0
            and gain_loss_ratio >= min_gain_loss_ratio
        )
        if enabled:
            allowed_families.append(fam)
        family_rows.append(
            {
                "family": fam,
                "total": total,
                "raw_correct": raw_correct,
                "raw_rate": raw_correct / total if total else 0.0,
                "selected_correct": selected_correct,
                "selected_rate": selected_correct / total if total else 0.0,
                "delta_questions": delta,
                "delta_rate": delta / total if total else 0.0,
                "gains": gains,
                "losses": losses,
                "gain_loss_ratio": gain_loss_ratio,
                "enabled": enabled,
            }
        )

    return {
        "raw_results": raw_results,
        "selected_results": selected_results,
        "dataset": dataset_path,
        "min_delta_questions": min_delta_questions,
        "min_gain_loss_ratio": min_gain_loss_ratio,
        "allowed_families": allowed_families,
        "families": family_rows,
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _write_md(path: str, policy: Dict[str, object]) -> None:
    lines = [
        "# Family-Gated ML Selection Policy",
        "",
        f"Raw results: `{policy['raw_results']}`",
        f"Selected results: `{policy['selected_results']}`",
        f"Dataset: `{policy['dataset']}`",
        "",
        "The ML selector is enabled only for families where it improves the tune split.",
        "",
        "| Family | Enabled | Questions | Raw Top-1 | ML Top-1 | Delta | Gains | Losses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy["families"]:
        lines.append(
            f"| {row['family']} | {'yes' if row['enabled'] else 'no'} | {row['total']} | "
            f"{row['raw_correct']} ({_fmt_pct(float(row['raw_rate']))}) | "
            f"{row['selected_correct']} ({_fmt_pct(float(row['selected_rate']))}) | "
            f"{row['delta_questions']:+d} ({_fmt_pct(float(row['delta_rate']))}) | "
            f"{row['gains']} | {row['losses']} |"
        )
    lines.extend(
        [
            "",
            "Allowed families:",
            "",
            ", ".join(f"`{name}`" for name in policy["allowed_families"]) or "None",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive a conservative family-gated ML policy from a tune split."
    )
    parser.add_argument("--raw-results", required=True)
    parser.add_argument("--selected-results", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--min-delta-questions", type=int, default=1)
    parser.add_argument("--min-gain-loss-ratio", type=float, default=1.0)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    policy = derive_policy(
        raw_results=args.raw_results,
        selected_results=args.selected_results,
        dataset_path=args.dataset,
        min_delta_questions=args.min_delta_questions,
        min_gain_loss_ratio=args.min_gain_loss_ratio,
    )
    _write_json(args.out_json, policy)
    _write_md(args.out_md, policy)
    print("===== FAMILY-GATED ML POLICY =====")
    print(f"Allowed families: {', '.join(policy['allowed_families']) or 'none'}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
