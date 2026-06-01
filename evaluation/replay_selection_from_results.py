#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.qa import _select_best_candidate_semantic


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _query_key(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def replay_selection(
    results_path: str,
    use_stored_features: bool = False,
    include_failure_diagnostics: bool = False,
) -> Dict[str, object]:
    results = _load_json(results_path)
    details = list(results.get("details") or [])

    total = 0
    original_top1 = 0
    replay_top1 = 0
    any_correct = 0
    changed: List[Dict[str, object]] = []
    failure_diagnostics: List[Dict[str, object]] = []

    for detail in details:
        candidates = list(detail.get("candidates") or [])
        if not candidates:
            continue

        total += 1
        question = str(detail.get("effective_question") or detail.get("question") or "")
        original_label = str(candidates[0].get("label", "")).lower()
        original_top1 += int(original_label == "correct")
        any_correct += int(any(str(c.get("label", "")).lower() == "correct" for c in candidates))

        replay_candidates = []
        by_key = {}
        for cand in candidates:
            query = str(cand.get("query", "") or "")
            row = {
                "query": query,
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
            replay_candidates.append(row)
            by_key[_query_key(query)] = cand

        selected = _select_best_candidate_semantic(replay_candidates, question)
        selected_key = _query_key(str((selected or {}).get("query", "")))
        selected_original = by_key.get(selected_key, {})
        replay_label = str(selected_original.get("label", "")).lower()
        replay_top1 += int(replay_label == "correct")

        if include_failure_diagnostics and any(
            str(c.get("label", "")).lower() == "correct" for c in candidates
        ) and original_label != "correct":
            top_scored = max(
                replay_candidates,
                key=lambda row: float(row.get("selection_score", float("-inf"))),
            )
            top_scored_original = by_key.get(_query_key(str(top_scored.get("query", ""))), {})
            first_correct_original = next(
                (c for c in candidates if str(c.get("label", "")).lower() == "correct"),
                {},
            )
            first_correct_replay = next(
                (
                    row for row in replay_candidates
                    if _query_key(str(row.get("query", "")))
                    == _query_key(str(first_correct_original.get("query", "")))
                ),
                {},
            )
            failure_diagnostics.append(
                {
                    "id": detail.get("id"),
                    "question": detail.get("question"),
                    "original_label": original_label,
                    "replay_label": replay_label,
                    "selected_index": selected_original.get("index"),
                    "selected_score": (selected or {}).get("selection_score"),
                    "selected_shape_reasons": (selected or {}).get("selection_shape_reasons"),
                    "top_scored_label": str(top_scored_original.get("label", "")).lower(),
                    "top_scored_index": top_scored_original.get("index"),
                    "top_scored_score": top_scored.get("selection_score"),
                    "top_scored_shape_reasons": top_scored.get("selection_shape_reasons"),
                    "first_correct_index": first_correct_original.get("index"),
                    "first_correct_score": first_correct_replay.get("selection_score"),
                    "first_correct_shape_reasons": first_correct_replay.get("selection_shape_reasons"),
                    "first_correct_penalties": (
                        first_correct_replay.get("semantic_judge_report") or {}
                    ).get("penalties"),
                }
            )

        if _query_key(str(candidates[0].get("query", ""))) != selected_key:
            changed.append(
                {
                    "id": detail.get("id"),
                    "question": detail.get("question"),
                    "original_label": original_label,
                    "replay_label": replay_label,
                    "original_index": candidates[0].get("index"),
                    "replay_index": selected_original.get("index"),
                    "original_selection_score": candidates[0].get("selection_score"),
                    "replay_selection_score": (selected or {}).get("selection_score"),
                    "replay_score_breakdown": (selected or {}).get("selection_score_breakdown"),
                    "replay_coverage_missing": (selected or {}).get("coverage_missing"),
                    "replay_semantic_penalties": (
                        (selected or {}).get("semantic_judge_report") or {}
                    ).get("penalties"),
                }
            )

    return {
        "results_path": results_path,
        "total": total,
        "original_top1_correct": original_top1,
        "replay_top1_correct": replay_top1,
        "any_correct": any_correct,
        "changed_count": len(changed),
        "changed": changed,
        "failure_diagnostics": failure_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay current deterministic selector on an existing results JSON without LLM generation."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--enable-targeted-rescue",
        action="store_true",
        help="Enable experimental targeted rescue rules during replay only.",
    )
    parser.add_argument(
        "--use-stored-features",
        action="store_true",
        help="Use semantic/coverage reports stored in the results file instead of recomputing current features.",
    )
    parser.add_argument(
        "--failure-diagnostics",
        action="store_true",
        help="Print selector diagnostics for questions where a correct candidate exists but original top1 is wrong.",
    )
    args = parser.parse_args()

    if args.enable_targeted_rescue:
        os.environ["INFINEON_ENABLE_TARGETED_SELECTION_RESCUE"] = "1"

    report = replay_selection(
        args.results,
        use_stored_features=args.use_stored_features,
        include_failure_diagnostics=args.failure_diagnostics,
    )
    print("===== SELECTION REPLAY =====")
    print(f"Results: {report['results_path']}")
    print(f"Total: {report['total']}")
    print(f"Original top1 correct: {report['original_top1_correct']}")
    print(f"Replay top1 correct: {report['replay_top1_correct']}")
    print(
        "Replay delta: "
        f"{int(report['replay_top1_correct']) - int(report['original_top1_correct']):+d}"
    )
    print(f"Any correct: {report['any_correct']}")
    print(f"Changed selections: {report['changed_count']}")
    if report["changed"]:
        print("Changed IDs:")
        for row in report["changed"]:
            print(
                f"  {row['id']}: {row['original_label']} -> {row['replay_label']} "
                f"(candidate index {row['original_index']} -> {row['replay_index']})"
            )
    if args.failure_diagnostics:
        print("Failure diagnostics:")
        for row in report.get("failure_diagnostics") or []:
            print(
                f"  {row['id']}: replay={row['replay_label']} idx={row['selected_index']} "
                f"score={row['selected_score']} | top_scored={row['top_scored_label']} "
                f"idx={row['top_scored_index']} score={row['top_scored_score']} | "
                f"first_correct idx={row['first_correct_index']} "
                f"score={row['first_correct_score']} reasons={row['first_correct_shape_reasons']}"
            )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
