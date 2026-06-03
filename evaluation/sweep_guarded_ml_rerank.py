#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.apply_ml_ranker_to_results import apply_ml_ranker


def _parse_float_grid(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Grid must contain at least one numeric value.")
    return values


def _parse_int_grid(text: str) -> List[int]:
    values = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Grid must contain at least one integer value.")
    return values


def sweep(
    results_path: str,
    model_path: str,
    schema_path: str,
    margins: List[float],
    scores: List[float],
    ranks: List[int],
    structured_guard: bool = False,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for margin, score, max_rank in itertools.product(margins, scores, ranks):
        updated = apply_ml_ranker(
            results_path=results_path,
            model_path=model_path,
            schema_path=schema_path,
            guarded=True,
            min_margin=margin,
            min_score=score,
            max_rank=max_rank,
            structured_guard=structured_guard,
        )
        summary = updated["summary"]
        rewrite = updated["ml_rerank_rewrite"]
        rows.append(
            {
                "min_margin": margin,
                "min_score": score,
                "max_rank": max_rank,
                "changed_count": rewrite["changed_count"],
                "top1_correct": summary["top1_correct"],
                "top1_rate": summary["top1_correct_rate"],
                "any_correct": summary["any_correct"],
                "any_rate": summary["any_correct_rate"],
            }
        )

    rows.sort(
        key=lambda row: (
            float(row["top1_rate"]),
            float(row["any_rate"]),
            -int(row["changed_count"]),
            -float(row["min_margin"]),
            -float(row["min_score"]),
        ),
        reverse=True,
    )
    return {
        "results": results_path,
        "model": model_path,
        "schema": schema_path,
        "grid": {
            "min_margin": margins,
            "min_score": scores,
            "max_rank": ranks,
            "structured_guard": bool(structured_guard),
        },
        "best": rows[0] if rows else {},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune guarded ML reranking thresholds on a dev results file. "
            "Use the selected thresholds once on a separate test results file."
        )
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--margins", default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--scores", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--max-ranks", default="1,2,3,4,5,6,8")
    parser.add_argument(
        "--structured-guard",
        action="store_true",
        help="Tune guarded ML reranking with question/query contract safety checks.",
    )
    args = parser.parse_args()

    report = sweep(
        results_path=args.results,
        model_path=args.model,
        schema_path=args.schema,
        margins=_parse_float_grid(args.margins),
        scores=_parse_float_grid(args.scores),
        ranks=_parse_int_grid(args.max_ranks),
        structured_guard=args.structured_guard,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")

    best = report["best"]
    print("===== GUARDED ML RERANK SWEEP =====")
    print(f"Results: {args.results}")
    print(f"Model: {args.model}")
    print(f"Runs: {len(report['rows'])}")
    print(f"Structured guard: {'yes' if args.structured_guard else 'no'}")
    if best:
        print(
            "Best: "
            f"top1={best['top1_correct']} ({best['top1_rate']:.3f}), "
            f"any={best['any_correct']} ({best['any_rate']:.3f}), "
            f"changed={best['changed_count']}, "
            f"min_margin={best['min_margin']}, "
            f"min_score={best['min_score']}, "
            f"max_rank={best['max_rank']}"
        )
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
