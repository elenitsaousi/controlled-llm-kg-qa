#!/usr/bin/env python3
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.infineon_eval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run held-out Infineon KGQA evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="data/infineon/infineon_test_final.json",
    )
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", default="results/infineon_test_final_eval.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--llm", default="auto", choices=["auto", "infineon"])
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--query-timeout", type=float, default=None)
    parser.add_argument("--generation-runs", type=int, default=1)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--use-ml-ranking",
        action="store_true",
        help="Enable ML reranking. Query-plan guidance remains active through the pipeline.",
    )
    parser.add_argument(
        "--no-schema-ranking",
        action="store_true",
        help="Disable schema/intent candidate ranking when ML ranking is not used.",
    )
    parser.add_argument(
        "--ml-model",
        default="ranking/models/infineon_np_tfidf_ranker.json",
    )
    parser.add_argument(
        "--ambiguity-config",
        default="",
        help="Optional ambiguity config JSON for runtime regime prediction.",
    )
    args = parser.parse_args()

    results = evaluate(
        dataset_path=args.dataset,
        graph_path=args.graph,
        k=args.k,
        schema_path=args.schema,
        out_path=args.out,
        llm=args.llm,
        temperature=args.temperature,
        progress=args.progress,
        query_timeout=args.query_timeout,
        generation_runs=max(1, int(args.generation_runs)),
        use_ml_ranking=args.use_ml_ranking,
        use_schema_ranking=not args.no_schema_ranking,
        ml_model_path=args.ml_model,
        ambiguity_config_path=(args.ambiguity_config or None),
        enable_entity_linking=True,
    )

    summary = results["summary"]
    print("===== INFINEON HELD-OUT SUMMARY =====")
    print(f"Dataset: {args.dataset}")
    print(f"Total: {summary['total']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Gold invalid: {summary['gold_invalid']}")
    print(f"Gold timeout: {summary['gold_timeout']}")
    print(f"LLM generation failures: {summary.get('llm_generation_failures', 0)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
