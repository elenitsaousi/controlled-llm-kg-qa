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
        description="Run Infineon 100-question benchmark evaluation."
    )
    parser.add_argument(
        "--dataset",
        default="data/infineon/infineon_dataset_100.json",
    )
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--out", default="results/infineon_eval_100.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--llm",
        default="auto",
        choices=["auto", "infineon"],
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--query-timeout", type=float, default=None)
    parser.add_argument(
        "--generation-runs",
        type=int,
        default=1,
        help="How many independent LLM generation runs to merge per question.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--use-ml-ranking",
        action="store_true",
        help="Enable ML ranking (disabled by default to avoid train/test leakage).",
    )
    parser.add_argument(
        "--ml-model",
        default="ranking/models/infineon_np_tfidf_ranker.json",
    )
    parser.add_argument(
        "--ml-ambiguity-regimes",
        default="",
        help="Comma-separated ambiguity labels where ML is enabled (e.g. mid). Empty => ML for all.",
    )
    parser.add_argument(
        "--ambiguity-config",
        default="",
        help="Optional ambiguity config JSON (tau1/tau2 + entropy source) for runtime regime prediction.",
    )
    parser.add_argument(
        "--no-entity-linking",
        action="store_true",
        help="Disable entity canonicalization before candidate generation.",
    )
    parser.add_argument(
        "--entity-link-max-matches",
        type=int,
        default=5,
        help="Maximum entity mentions to canonicalize per question.",
    )
    args = parser.parse_args()

    ml_regimes = []
    for tok in args.ml_ambiguity_regimes.split(","):
        lab = tok.strip().lower()
        if not lab:
            continue
        if lab == "medium":
            lab = "mid"
        ml_regimes.append(lab)
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
        ml_model_path=args.ml_model,
        ml_ambiguity_regimes=ml_regimes,
        ambiguity_config_path=(args.ambiguity_config or None),
        enable_entity_linking=not args.no_entity_linking,
        entity_link_max_matches=max(1, int(args.entity_link_max_matches)),
    )

    summary = results["summary"]
    print("===== INFINEON-100 SUMMARY =====")
    print(f"Total: {summary['total']}")
    print(f"Top1 correct: {summary['top1_correct']} ({summary['top1_correct_rate']:.3f})")
    print(f"Any correct: {summary['any_correct']} ({summary['any_correct_rate']:.3f})")
    print(f"Gold invalid: {summary['gold_invalid']}")
    print(f"Gold timeout: {summary['gold_timeout']}")
    print(f"LLM generation failures: {summary.get('llm_generation_failures', 0)}")
    print(
        "Repairs: "
        f"enabled={summary.get('repair_enabled', False)} "
        f"attempted={summary.get('repair_candidates_attempted', 0)} "
        f"succeeded={summary.get('repair_candidates_succeeded', 0)}"
    )
    print(f"ML ranking: {summary['ml_ranking']}")
    print(
        "Entity linking: "
        f"enabled={summary.get('entity_linking_enabled', False)} "
        f"changed_questions={summary.get('entity_linked_questions', 0)}/{summary['total']}"
    )
    if summary.get("ml_ambiguity_regimes"):
        print(f"ML ambiguity regimes: {','.join(summary['ml_ambiguity_regimes'])}")
    if summary.get("ambiguity_config_path"):
        print(f"Ambiguity config: {summary['ambiguity_config_path']}")
        if summary.get("predicted_regime_counts"):
            print(f"Predicted regimes: {summary['predicted_regime_counts']}")

    per_amb = summary.get("per_ambiguity", {})
    if per_amb:
        print("\nPer ambiguity label:")
        for label in ("low", "mid", "high"):
            if label not in per_amb:
                continue
            s = per_amb[label]
            print(
                f"  {label}: n={s['total']} "
                f"top1={s['top1_correct']}/{s['total']} ({s['top1_correct_rate']:.3f}) "
                f"any={s['any_correct']}/{s['total']} ({s['any_correct_rate']:.3f})"
            )

    print(f"\nSaved results to: {args.out}")


if __name__ == "__main__":
    main()
