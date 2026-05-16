#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kg.schema import load_schema
from llm.client import InfineonGPTClient
from pipeline.qa import answer_question


def run_eval(
    dataset_path: str,
    schema_path: str,
    *,
    use_ml_ranking: bool,
    ml_policy: str,
    ml_model_path: str,
    progress: bool,
) -> Dict[str, object]:
    rows = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    schema = load_schema(schema_path)
    client = InfineonGPTClient()
    details: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        if progress:
            print(f"[{idx}/{len(rows)}] {row['id']} - {row['question']}")
        result = answer_question(
            str(row["question"]),
            schema,
            llm_client=client,
            use_ml_ranking=use_ml_ranking,
            ml_policy=ml_policy,
            ml_model_path=ml_model_path or None,
            include_candidate_diagnostics=False,
        )
        details.append(
            {
                "id": row["id"],
                "question": row["question"],
                "topic": row.get("topic"),
                "expected_needs_clarification": row["expected_needs_clarification"],
                "clarification": result.get("clarification"),
                "request_route": result.get("request_route"),
                "selected_query": result.get("selected_query"),
                "candidate_count": len(result.get("candidates") or []),
                "candidates": result.get("candidates") or [],
            }
        )
    return {"details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live clarification-behavior QA evaluation.")
    parser.add_argument("--dataset", default="data/infineon/clarification_behavior_eval.json")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--use-ml-ranking", action="store_true")
    parser.add_argument("--ml-policy", default="auto")
    parser.add_argument("--ml-model", default="")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    report = run_eval(
        args.dataset,
        args.schema,
        use_ml_ranking=bool(args.use_ml_ranking),
        ml_policy=args.ml_policy,
        ml_model_path=args.ml_model,
        progress=bool(args.progress),
    )
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("===== CLARIFICATION BEHAVIOR RUN =====")
    print(f"Dataset: {args.dataset}")
    print(f"Total: {len(report['details'])}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
