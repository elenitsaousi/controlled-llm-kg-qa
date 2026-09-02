#!/usr/bin/env python3
"""Paired bootstrap CI for Top-1 accuracy between two selection results.

Compares per-question top1_correct flags from two result files that share
the same question ids (e.g. raw LLM order vs guarded ML reranking on the
repaired 1000-question KG selection benchmark, Table 6.1/6.3). Reports:

  - accuracy and a percentile bootstrap 95% CI for each result file,
  - a paired bootstrap 95% CI for the accuracy difference (b - a),
  - an exact two-sided McNemar test on the discordant pairs, as a
    standard complementary significance check for paired binary outcomes.

Resampling is over questions (not candidates), preserving the pairing
between the two result files on each bootstrap draw.
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple


def _load_top1(path: str) -> Dict[str, bool]:
    payload = json.load(open(path, "r", encoding="utf-8"))
    details = payload.get("details") or payload.get("rows") or []
    out: Dict[str, bool] = {}
    for row in details:
        if not isinstance(row, dict):
            continue
        qid = row.get("id") or row.get("question_id") or row.get("request_id")
        if qid is None:
            continue
        out[str(qid)] = bool(row.get("top1_correct"))
    return out


def _percentile(sorted_values: List[float], p: float) -> float:
    idx = int(round(p * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    return min(1.0, 2 * cdf)


def analyze(
    raw_path: str,
    ml_path: str,
    *,
    label_a: str,
    label_b: str,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    a_map = _load_top1(raw_path)
    b_map = _load_top1(ml_path)
    ids = sorted(set(a_map) & set(b_map))
    if not ids:
        raise SystemExit("No overlapping question ids between the two result files.")
    missing_a = set(b_map) - set(a_map)
    missing_b = set(a_map) - set(b_map)

    pairs: List[Tuple[bool, bool]] = [(a_map[i], b_map[i]) for i in ids]
    n = len(pairs)

    a_correct = sum(1 for a, _ in pairs if a)
    b_correct = sum(1 for _, b in pairs if b)
    a_acc = a_correct / n
    b_acc = b_correct / n
    observed_diff = b_acc - a_acc

    b_only = sum(1 for a, b in pairs if not a and b)  # b correct, a wrong
    a_only = sum(1 for a, b in pairs if a and not b)  # a correct, b wrong
    mcnemar_p = _mcnemar_exact_p(a_only, b_only)

    rng = random.Random(seed)
    diffs: List[float] = []
    a_accs: List[float] = []
    b_accs: List[float] = []
    for _ in range(bootstrap_samples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        sa = sum(1 for a, _ in sample if a) / n
        sb = sum(1 for _, b in sample if b) / n
        a_accs.append(sa)
        b_accs.append(sb)
        diffs.append(sb - sa)

    diffs.sort()
    a_accs.sort()
    b_accs.sort()

    ci_diff = (_percentile(diffs, 0.025), _percentile(diffs, 0.975))
    ci_a = (_percentile(a_accs, 0.025), _percentile(a_accs, 0.975))
    ci_b = (_percentile(b_accs, 0.025), _percentile(b_accs, 0.975))
    prob_diff_gt0 = sum(1 for d in diffs if d > 0) / bootstrap_samples

    return {
        "raw_path": raw_path,
        "ml_path": ml_path,
        "label_a": label_a,
        "label_b": label_b,
        "n": n,
        "missing_ids_only_in_b": sorted(missing_a),
        "missing_ids_only_in_a": sorted(missing_b),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "a_correct": a_correct,
        "b_correct": b_correct,
        "a_accuracy": a_acc,
        "b_accuracy": b_acc,
        "observed_diff": observed_diff,
        "bootstrap_ci_a": ci_a,
        "bootstrap_ci_b": ci_b,
        "bootstrap_ci_diff": ci_diff,
        "bootstrap_prob_diff_gt_zero": prob_diff_gt0,
        "mcnemar_a_only_correct": a_only,
        "mcnemar_b_only_correct": b_only,
        "mcnemar_exact_p_value": mcnemar_p,
    }


def _write_md(report: Dict[str, object], out_md: str) -> None:
    a, b = report["label_a"], report["label_b"]
    lines = [
        "# Paired Bootstrap CI: Top-1 Accuracy",
        "",
        f"Comparing **{a}** vs **{b}** on {report['n']} shared questions "
        f"({report['bootstrap_samples']} bootstrap resamples, seed={report['seed']}).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| {a} accuracy | {report['a_accuracy']:.4f} ({report['a_correct']}/{report['n']}) |",
        f"| {b} accuracy | {report['b_accuracy']:.4f} ({report['b_correct']}/{report['n']}) |",
        f"| Observed diff ({b} - {a}) | {report['observed_diff']:.4f} |",
        f"| Bootstrap 95% CI, {a} | [{report['bootstrap_ci_a'][0]:.4f}, {report['bootstrap_ci_a'][1]:.4f}] |",
        f"| Bootstrap 95% CI, {b} | [{report['bootstrap_ci_b'][0]:.4f}, {report['bootstrap_ci_b'][1]:.4f}] |",
        f"| Bootstrap 95% CI, diff | [{report['bootstrap_ci_diff'][0]:.4f}, {report['bootstrap_ci_diff'][1]:.4f}] |",
        f"| P(bootstrap diff > 0) | {report['bootstrap_prob_diff_gt_zero']:.4f} |",
        f"| McNemar discordant pairs | {a}-only correct={report['mcnemar_a_only_correct']}, {b}-only correct={report['mcnemar_b_only_correct']} |",
        f"| McNemar exact two-sided p-value | {report['mcnemar_exact_p_value']:.5f} |",
        "",
    ]
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap CI and McNemar test comparing Top-1 accuracy between two result files."
    )
    parser.add_argument("--raw", required=True, help="First result file (details[].id, details[].top1_correct)")
    parser.add_argument("--ml", required=True, help="Second result file (same shape, same question ids)")
    parser.add_argument("--label-a", default="raw", help="Label for --raw in the report")
    parser.add_argument("--label-b", default="guarded_ml", help="Label for --ml in the report")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(
        args.raw,
        args.ml,
        label_a=args.label_a,
        label_b=args.label_b,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(report, args.out_md)

    print("===== PAIRED BOOTSTRAP TOP-1 ACCURACY CI =====")
    print(f"n = {report['n']}")
    print(f"{args.label_a}: {report['a_correct']}/{report['n']} = {report['a_accuracy']:.4f}")
    print(f"{args.label_b}: {report['b_correct']}/{report['n']} = {report['b_accuracy']:.4f}")
    print(f"observed diff = {report['observed_diff']:.4f}")
    print(f"bootstrap 95% CI diff = [{report['bootstrap_ci_diff'][0]:.4f}, {report['bootstrap_ci_diff'][1]:.4f}]")
    print(f"P(bootstrap diff > 0) = {report['bootstrap_prob_diff_gt_zero']:.4f}")
    print(
        f"McNemar: a_only={report['mcnemar_a_only_correct']} b_only={report['mcnemar_b_only_correct']} "
        f"p={report['mcnemar_exact_p_value']:.5f}"
    )
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
