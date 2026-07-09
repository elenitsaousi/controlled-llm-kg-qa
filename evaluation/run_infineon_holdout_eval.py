#!/usr/bin/env python3
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.infineon_eval import EvaluationAbortedError, evaluate, _parse_amb_regimes


def _detail_count(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return 0
    details = payload.get("details") if isinstance(payload, dict) else None
    return len(details) if isinstance(details, list) else 0


def _choose_resume_path(out_path: str, resume_from: str) -> str:
    explicit = resume_from or ""
    if not explicit:
        return out_path
    explicit_count = _detail_count(explicit)
    out_count = _detail_count(out_path)
    if out_count > explicit_count:
        print(
            "Resume safety: --out contains more completed rows than --resume-from "
            f"({out_count} > {explicit_count}); resuming from --out instead.",
            flush=True,
        )
        return out_path
    return explicit


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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the existing --out file by skipping completed question IDs.",
    )
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional existing result JSON to resume from. Defaults to --out when --resume is set.",
    )
    parser.add_argument(
        "--skip-ids",
        nargs="*",
        default=[],
        help="Question IDs to mark as skipped/failure and continue, e.g. FINALKGQA300.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--keep-going-on-auth-error",
        action="store_true",
        help="Treat SSO/auth redirects as ordinary LLM failures instead of aborting immediately.",
    )
    parser.add_argument(
        "--use-ml-ranking",
        action="store_true",
        help="Enable ML reranking. Query-plan guidance remains active through the pipeline.",
    )
    parser.add_argument(
        "--use-schema-ranking",
        action="store_true",
        help="Enable schema/intent candidate ranking when ML ranking is not used.",
    )
    parser.add_argument(
        "--use-semantic-selection",
        action="store_true",
        help="Enable conservative semantic candidate selection when ML/schema ranking is not used.",
    )
    parser.add_argument(
        "--semantic-selection-margin",
        type=float,
        default=1.25,
        help="Minimum semantic score margin required to override the first candidate.",
    )
    parser.add_argument(
        "--ml-model",
        default="ranking/models/infineon_np_tfidf_ranker.json",
    )
    parser.add_argument(
        "--ml-ambiguity-regimes",
        default="",
        help="Comma-separated ambiguity labels where ML ranking is used, e.g. low,mid.",
    )
    parser.add_argument(
        "--ambiguity-config",
        default="",
        help="Optional ambiguity config JSON for runtime regime prediction.",
    )
    args = parser.parse_args()

    try:
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
            use_schema_ranking=args.use_schema_ranking,
            use_semantic_selection=args.use_semantic_selection,
            semantic_selection_margin=args.semantic_selection_margin,
            ml_model_path=args.ml_model,
            ml_ambiguity_regimes=_parse_amb_regimes(args.ml_ambiguity_regimes),
            ambiguity_config_path=(args.ambiguity_config or None),
            enable_entity_linking=True,
            fail_on_auth_error=not args.keep_going_on_auth_error,
            limit=args.limit,
            resume_path=_choose_resume_path(args.out, args.resume_from) if args.resume else None,
            skip_ids=args.skip_ids,
        )
    except EvaluationAbortedError as exc:
        print(f"ABORTED: {exc}")
        print(f"Partial output: {args.out}")
        raise SystemExit(2) from exc

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
