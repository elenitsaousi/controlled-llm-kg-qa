#!/usr/bin/env python3
"""
Orchestrate the first 5 Infineon pipeline steps end-to-end.

Steps:
1) Generate 500-question dataset
2) Create train/dev/test split
3) Build candidate pools for train/dev/test
4) Train ranker + split KPI report
5) Fit ambiguity policy config on dev
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Infineon 500 pipeline (steps 1-5)."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset-100", default="data/infineon/infineon_dataset_100.json")
    parser.add_argument("--dataset-500", default="data/infineon/infineon_dataset_500.json")
    parser.add_argument("--splits-dir", default="data/infineon/splits/infineon_500")
    parser.add_argument("--schema", default="data/infineon/schema.json")
    parser.add_argument("--graph", default="data/infineon/graph.ttl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-ambiguity", action="store_true")
    args = parser.parse_args()

    py = args.python
    splits_dir = Path(args.splits_dir)
    train_split = str(splits_dir / "train.json")
    dev_split = str(splits_dir / "dev.json")
    test_split = str(splits_dir / "test.json")

    train_candidates = "ranking/infineon_train_candidates_500.json"
    dev_candidates = "ranking/infineon_dev_candidates_500.json"
    test_candidates = "ranking/infineon_test_candidates_500.json"
    model_path = "ranking/models/infineon_np_tfidf_ranker_500.json"
    split_report = "results/infineon_split_kpi_500.json"
    ambiguity_cfg = "ranking/models/infineon_ambiguity_config_500.json"
    ambiguity_report = "results/infineon_ambiguity_calibration_500.json"
    kpi_summary = "results/infineon_kpi_summary_500.json"

    if not args.skip_generate:
        _run(
            [
                py,
                "data/infineon/generate_infineon_dataset_500.py",
                "--seed",
                args.dataset_100,
                "--out",
                args.dataset_500,
                "--target",
                "500",
            ]
        )

    if not args.skip_split:
        _run(
            [
                py,
                "data/infineon/split_infineon_dataset.py",
                "--dataset",
                args.dataset_500,
                "--out-dir",
                args.splits_dir,
                "--ratios",
                "0.8,0.1,0.1",
                "--seed",
                "42",
            ]
        )

    if not args.skip_build:
        for split_file, out_file in (
            (train_split, train_candidates),
            (dev_split, dev_candidates),
            (test_split, test_candidates),
        ):
            _run(
                [
                    py,
                    "ranking/build_infineon_training_data.py",
                    "--dataset",
                    split_file,
                    "--graph",
                    args.graph,
                    "--schema",
                    args.schema,
                    "--out",
                    out_file,
                    "--k",
                    str(args.k),
                    "--n-runs",
                    str(args.n_runs),
                ]
            )

    if not args.skip_train:
        _run(
            [
                py,
                "ranking/train_infineon_np_tfidf_split.py",
                "--train-data",
                train_candidates,
                "--dev-data",
                dev_candidates,
                "--test-data",
                test_candidates,
                "--model-out",
                model_path,
                "--report-out",
                split_report,
                "--ml-regimes",
                "mid",
            ]
        )

    if not args.skip_ambiguity:
        _run(
            [
                py,
                "analysis/fit_infineon_ambiguity_policy.py",
                "--train-data",
                train_candidates,
                "--calib-data",
                dev_candidates,
                "--model",
                model_path,
                "--schema",
                args.schema,
                "--graph",
                args.graph,
                "--entropy-source",
                "agreement",
                "--ml-regimes",
                "mid",
                "--out-config",
                ambiguity_cfg,
                "--out-report",
                ambiguity_report,
            ]
        )

    _run(
        [
            py,
            "analysis/infineon_kpi_report.py",
            "--split-report",
            split_report,
            "--out",
            kpi_summary,
        ]
    )

    print("\nPipeline completed.")
    print(f"Model: {model_path}")
    print(f"Ambiguity config: {ambiguity_cfg}")
    print(f"KPI summary: {kpi_summary}")


if __name__ == "__main__":
    main()
