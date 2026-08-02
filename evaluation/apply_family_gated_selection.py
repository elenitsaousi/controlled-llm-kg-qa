#!/usr/bin/env python3
import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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


def _details_by_id(payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
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


def _recompute_summary(details: List[Dict[str, object]], original_summary: Dict[str, object]) -> Dict[str, object]:
    summary = deepcopy(original_summary)
    keys_to_zero = [
        "total",
        "top1_correct",
        "top1_valid_wrong",
        "top1_invalid",
        "any_correct",
        "total_candidates",
        "correct_candidates",
        "valid_wrong_candidates",
        "invalid_candidates",
        "candidate_timeouts",
        "all_invalid",
        "all_valid_wrong",
        "llm_generation_failures",
    ]
    for key in keys_to_zero:
        summary[key] = 0
    summary["per_ambiguity"] = {}

    for detail in details:
        summary["total"] += 1
        amb = str(detail.get("ambiguity_label") or "").strip()
        amb_summary = None
        if amb:
            amb_summary = summary["per_ambiguity"].setdefault(
                amb,
                {
                    "total": 0,
                    "gold_invalid": 0,
                    "gold_timeout": 0,
                    "top1_correct": 0,
                    "any_correct": 0,
                },
            )
            amb_summary["total"] += 1

        candidates = detail.get("candidates") or []
        if detail.get("generation_error"):
            summary["llm_generation_failures"] += 1
        if not candidates:
            summary["top1_invalid"] += 1
            summary["all_invalid"] += 1
            continue

        top_label = str(candidates[0].get("label", "")).lower()
        any_valid = False
        any_correct = False
        for cand in candidates:
            label = str(cand.get("label", "")).lower()
            summary["total_candidates"] += 1
            if label == "correct":
                summary["correct_candidates"] += 1
                any_valid = True
                any_correct = True
            elif label == "valid_wrong":
                summary["valid_wrong_candidates"] += 1
                any_valid = True
            elif label == "timeout":
                summary["invalid_candidates"] += 1
                summary["candidate_timeouts"] += 1
            elif label == "invalid":
                summary["invalid_candidates"] += 1

        if top_label == "correct":
            summary["top1_correct"] += 1
        elif top_label == "valid_wrong":
            summary["top1_valid_wrong"] += 1
        else:
            summary["top1_invalid"] += 1
        if any_correct:
            summary["any_correct"] += 1
        if not any_valid:
            summary["all_invalid"] += 1
        if any_valid and not any_correct:
            summary["all_valid_wrong"] += 1
        detail["top1_correct"] = top_label == "correct"
        detail["any_correct"] = any_correct
        if amb_summary is not None:
            amb_summary["top1_correct"] += int(top_label == "correct")
            amb_summary["any_correct"] += int(any_correct)

    denom = summary["total"] or 1
    summary["top1_correct_rate"] = summary["top1_correct"] / denom
    summary["any_correct_rate"] = summary["any_correct"] / denom
    total_candidates = summary["total_candidates"] or 1
    summary["candidate_correct_rate"] = summary["correct_candidates"] / total_candidates
    summary["candidate_invalid_rate"] = summary["invalid_candidates"] / total_candidates
    for stats in summary["per_ambiguity"].values():
        amb_denom = stats["total"] or 1
        stats["top1_correct_rate"] = stats["top1_correct"] / amb_denom
        stats["any_correct_rate"] = stats["any_correct"] / amb_denom
    return summary


def apply_family_gate(
    *,
    raw_results_path: str,
    selected_results_path: str,
    dataset_path: str,
    policy_path: str,
) -> Dict[str, object]:
    raw_payload = _load_json(raw_results_path)
    selected_payload = _load_json(selected_results_path)
    policy = _load_json(policy_path)
    dataset = _dataset_by_id(dataset_path)
    selected_by_id = _details_by_id(selected_payload)
    allowed = set(str(name) for name in policy.get("allowed_families") or [])

    details = []
    changed = []
    for raw_detail in list(raw_payload.get("details") or []):
        if not isinstance(raw_detail, dict):
            continue
        qid = str(raw_detail.get("id", "")).strip()
        fam = _family(qid, dataset, raw_detail)
        selected_detail = selected_by_id.get(qid)
        use_selected = fam in allowed and selected_detail is not None
        detail = deepcopy(selected_detail if use_selected else raw_detail)
        detail["family_gated_selection"] = {
            "family": fam,
            "used_ml_selection": bool(use_selected),
            "policy": policy_path,
        }
        details.append(detail)
        if use_selected:
            raw_label = (
                str((raw_detail.get("candidates") or [{}])[0].get("label", "")).lower()
                if raw_detail.get("candidates")
                else ""
            )
            selected_label = (
                str((selected_detail.get("candidates") or [{}])[0].get("label", "")).lower()
                if selected_detail.get("candidates")
                else ""
            )
            changed.append(
                {
                    "id": qid,
                    "family": fam,
                    "raw_label": raw_label,
                    "selected_label": selected_label,
                    "raw_top1_correct": bool(raw_detail.get("top1_correct")),
                    "selected_top1_correct": bool(selected_detail.get("top1_correct")),
                }
            )

    output = deepcopy(raw_payload)
    output["details"] = details
    output["summary"] = _recompute_summary(details, raw_payload.get("summary") or {})
    output["family_gated_selection"] = {
        "raw_results": raw_results_path,
        "selected_results": selected_results_path,
        "dataset": dataset_path,
        "policy": policy_path,
        "allowed_families": sorted(allowed),
        "ml_applied_count": len(changed),
        "changed": changed,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply ML-selected results only to families enabled by a tune-derived policy."
    )
    parser.add_argument("--raw-results", required=True)
    parser.add_argument("--selected-results", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = apply_family_gate(
        raw_results_path=args.raw_results,
        selected_results_path=args.selected_results,
        dataset_path=args.dataset,
        policy_path=args.policy,
    )
    _write_json(args.out, output)
    summary = output["summary"]
    gate = output["family_gated_selection"]
    print("===== APPLY FAMILY-GATED SELECTION =====")
    print(f"Allowed families: {', '.join(gate['allowed_families']) or 'none'}")
    print(f"ML applied count: {gate['ml_applied_count']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
