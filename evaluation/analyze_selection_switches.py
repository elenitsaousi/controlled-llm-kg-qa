#!/usr/bin/env python3
"""Compare two result files and audit selection switches.

Use this to understand whether a reranker improves selection by making good
switches, loses accuracy by switching away from correct top1 candidates, or
misses cases where a correct candidate was available.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "") or "").lower()


def _top(detail: Dict[str, object]) -> Dict[str, object]:
    candidates = detail.get("candidates") or []
    return candidates[0] if candidates else {}


def _first_correct_idx(detail: Dict[str, object]) -> int:
    for idx, candidate in enumerate(detail.get("candidates") or []):
        if _label(candidate) == "correct":
            return idx
    return -1


def _first_correct_candidate(detail: Dict[str, object]) -> Dict[str, object]:
    idx = _first_correct_idx(detail)
    candidates = detail.get("candidates") or []
    if idx < 0 or idx >= len(candidates):
        return {}
    return candidates[idx]


def _source(candidate: Dict[str, object]) -> str:
    return str(candidate.get("source") or "unknown")


def _family(detail: Dict[str, object], dataset_by_id: Dict[str, Dict[str, object]]) -> str:
    qid = str(detail.get("id") or "")
    row = dataset_by_id.get(qid, {})
    return str(detail.get("family") or row.get("family") or "unknown")


def _dataset_by_id(path: str | None) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}
    raw = _load_json(path)
    if not isinstance(raw, list):
        return {}
    return {str(row.get("id") or ""): row for row in raw if isinstance(row, dict)}


def analyze(
    before_path: str,
    after_path: str,
    dataset_path: str | None = None,
) -> Dict[str, object]:
    before_payload = _load_json(before_path)
    after_payload = _load_json(after_path)
    dataset = _dataset_by_id(dataset_path)

    before_details = {
        str(detail.get("id") or ""): detail
        for detail in before_payload.get("details", [])
        if isinstance(detail, dict)
    }
    after_details = {
        str(detail.get("id") or ""): detail
        for detail in after_payload.get("details", [])
        if isinstance(detail, dict)
    }

    rows: List[Dict[str, object]] = []
    counts = Counter()
    family_counts = Counter()
    source_moves = Counter()
    missed = []

    for qid, before in before_details.items():
        after = after_details.get(qid)
        if not after:
            continue
        before_top = _top(before)
        after_top = _top(after)
        before_key = _query_key(str(before_top.get("query", "")))
        after_key = _query_key(str(after_top.get("query", "")))
        before_label = _label(before_top)
        after_label = _label(after_top)
        any_correct = _first_correct_idx(before) >= 0
        changed = before_key != after_key
        family = _family(before, dataset)

        if not changed and any_correct and before_label != "correct":
            correct = _first_correct_candidate(before)
            missed.append(
                {
                    "id": qid,
                    "family": family,
                    "first_correct_rank": _first_correct_idx(before) + 1,
                    "top_source": _source(before_top),
                    "top_label": before_label,
                    "correct_source": _source(correct),
                    "top_ml_score": before_top.get("ml_score"),
                    "correct_ml_score": correct.get("ml_score"),
                    "top_selection_score": before_top.get("selection_score"),
                    "correct_selection_score": correct.get("selection_score"),
                    "question": before.get("question") or before.get("effective_question"),
                }
            )

        if not changed:
            counts["unchanged"] += 1
            continue

        if before_label != "correct" and after_label == "correct":
            outcome = "good_switch"
        elif before_label == "correct" and after_label != "correct":
            outcome = "bad_switch_lost_correct"
        elif before_label != "correct" and after_label != "correct":
            outcome = "wrong_to_wrong_switch"
        else:
            outcome = "correct_to_correct_switch"

        counts[outcome] += 1
        family_counts[(outcome, family)] += 1
        source_moves[(_source(before_top), _source(after_top), outcome)] += 1
        rows.append(
            {
                "id": qid,
                "family": family,
                "outcome": outcome,
                "before_label": before_label,
                "after_label": after_label,
                "before_source": _source(before_top),
                "after_source": _source(after_top),
                "first_correct_rank": _first_correct_idx(before) + 1,
                "question": before.get("question") or before.get("effective_question"),
            }
        )

    rows.sort(key=lambda r: (str(r["outcome"]), str(r["family"]), str(r["id"])))
    missed.sort(key=lambda r: (str(r["family"]), int(r["first_correct_rank"]), str(r["id"])))

    return {
        "before": before_path,
        "after": after_path,
        "summary": {
            "changed": len(rows),
            "unchanged": counts["unchanged"],
            "good_switch": counts["good_switch"],
            "bad_switch_lost_correct": counts["bad_switch_lost_correct"],
            "wrong_to_wrong_switch": counts["wrong_to_wrong_switch"],
            "correct_to_correct_switch": counts["correct_to_correct_switch"],
            "missed_with_correct_candidate": len(missed),
        },
        "top_family_outcomes": [
            {"outcome": outcome, "family": family, "count": count}
            for (outcome, family), count in family_counts.most_common(30)
        ],
        "top_source_moves": [
            {"from": src_from, "to": src_to, "outcome": outcome, "count": count}
            for (src_from, src_to, outcome), count in source_moves.most_common(30)
        ],
        "switches": rows,
        "missed": missed,
    }


def _write_md(report: Dict[str, object], out_md: str) -> None:
    summary = report["summary"]
    lines = [
        "# Selection Switch Audit",
        "",
        f"- Before: `{report['before']}`",
        f"- After: `{report['after']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Family Outcomes", ""])
    for row in report["top_family_outcomes"]:
        lines.append(f"- {row['outcome']} / {row['family']}: {row['count']}")
    lines.extend(["", "## Top Source Moves", ""])
    for row in report["top_source_moves"]:
        lines.append(f"- {row['from']} -> {row['to']} / {row['outcome']}: {row['count']}")
    lines.extend(["", "## Switches", ""])
    for row in report["switches"][:120]:
        lines.append(
            f"- `{row['id']}` {row['outcome']} [{row['family']}]: "
            f"{row['before_label']}({row['before_source']}) -> "
            f"{row['after_label']}({row['after_source']}), "
            f"first_correct_rank={row['first_correct_rank']}"
        )
    lines.extend(["", "## Missed With Correct Candidate", ""])
    for row in report["missed"][:120]:
        lines.append(
            f"- `{row['id']}` [{row['family']}]: top={row['top_label']} "
            f"source={row['top_source']}, first_correct_rank={row['first_correct_rank']}, "
            f"correct_source={row.get('correct_source')}, "
            f"top_ml={row.get('top_ml_score')}, correct_ml={row.get('correct_ml_score')}"
        )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit selection switches between two KGQA result files.")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(args.before, args.after, args.dataset)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(report, args.out_md)

    s = report["summary"]
    print("===== SELECTION SWITCH AUDIT =====")
    print(f"Before: {args.before}")
    print(f"After: {args.after}")
    print(f"Changed: {s['changed']}")
    print(f"Good switches: {s['good_switch']}")
    print(f"Bad switches lost correct: {s['bad_switch_lost_correct']}")
    print(f"Wrong-to-wrong switches: {s['wrong_to_wrong_switch']}")
    print(f"Missed with correct candidate: {s['missed_with_correct_candidate']}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
