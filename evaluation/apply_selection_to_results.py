#!/usr/bin/env python3
import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.qa import _select_best_candidate_semantic


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _selector_payload(cand: Dict[str, object], use_stored_features: bool) -> Dict[str, object]:
    row = {
        "query": str(cand.get("query", "") or ""),
        "ml_score": cand.get("ml_score"),
        "source": cand.get("source"),
        "validated_retrieval_score": cand.get("validated_retrieval_score"),
        "validated_source": cand.get("validated_source"),
    }
    if use_stored_features:
        row.update(
            {
                "semantic_judge_score": cand.get("semantic_judge_score"),
                "semantic_judge_report": cand.get("semantic_judge_report"),
                "coverage_score": cand.get("coverage_score"),
                "coverage_missing": cand.get("coverage_missing"),
                "coverage_required": cand.get("coverage_required"),
            }
        )
    return row


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


def apply_selection(results_path: str, use_stored_features: bool) -> Dict[str, object]:
    payload = _load_json(results_path)
    details = deepcopy(payload.get("details") or [])
    changed = []

    for detail in details:
        candidates = list(detail.get("candidates") or [])
        if not candidates:
            continue
        question = str(detail.get("effective_question") or detail.get("question") or "")
        selector_rows = [_selector_payload(c, use_stored_features) for c in candidates]
        selected = _select_best_candidate_semantic(selector_rows, question)
        if not selected:
            continue
        selected_key = _query_key(str(selected.get("query", "")))
        selected_idx = next(
            (idx for idx, cand in enumerate(candidates) if _query_key(str(cand.get("query", ""))) == selected_key),
            None,
        )
        if selected_idx is None or selected_idx == 0:
            continue

        before_label = str(candidates[0].get("label", "")).lower()
        chosen = candidates[selected_idx]
        after_label = str(chosen.get("label", "")).lower()
        detail["candidates"] = [chosen] + candidates[:selected_idx] + candidates[selected_idx + 1:]
        changed.append(
            {
                "id": detail.get("id"),
                "from_label": before_label,
                "to_label": after_label,
                "from_index": candidates[0].get("index"),
                "to_index": chosen.get("index"),
            }
        )

    payload["summary"] = _recompute_summary(details, payload.get("summary") or {})
    payload["details"] = details
    payload["selection_rewrite"] = {
        "source_results": results_path,
        "changed_count": len(changed),
        "changed": changed,
        "use_stored_features": bool(use_stored_features),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the current deterministic selector to an existing evaluation JSON without new LLM calls."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--enable-targeted-rescue", action="store_true")
    parser.add_argument("--use-stored-features", action="store_true")
    args = parser.parse_args()

    if args.enable_targeted_rescue:
        os.environ["INFINEON_ENABLE_TARGETED_SELECTION_RESCUE"] = "1"

    updated = apply_selection(args.results, use_stored_features=args.use_stored_features)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    summary = updated["summary"]
    rewrite = updated["selection_rewrite"]
    print("===== APPLY SELECTION TO RESULTS =====")
    print(f"Input: {args.results}")
    print(f"Changed selections: {rewrite['changed_count']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
