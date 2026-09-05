#!/usr/bin/env python3
"""Create an SVG line chart for baseline vs ML accuracy by entropy regime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_report(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Report JSON must be an object.")
    return payload


def _regime_rows(report: Dict[str, object]) -> List[Dict[str, object]]:
    rows = report.get("by_entropy_regime") or []
    if not isinstance(rows, list):
        return []
    by_name = {str(row.get("regime")): row for row in rows if isinstance(row, dict)}
    return [dict(by_name.get(name) or {"regime": name}) for name in ("low", "medium", "high")]


def _y(value: float, top: float = 170.0, height: float = 360.0) -> float:
    value = max(0.0, min(1.0, value))
    return top + (1.0 - value) * height


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_svg(report: Dict[str, object], title: str, subtitle: str) -> str:
    rows = _regime_rows(report)
    xs = [256.0, 560.0, 864.0]
    baseline = [float(row.get("baseline_accuracy") or 0.0) for row in rows]
    ml = [float(row.get("ml_accuracy") or 0.0) for row in rows]
    baseline_points = " ".join(f"{x:.0f},{_y(v):.2f}" for x, v in zip(xs, baseline))
    ml_points = " ".join(f"{x:.0f},{_y(v):.2f}" for x, v in zip(xs, ml))
    summary = dict(report.get("summary") or {})
    baseline_total = float(summary.get("baseline_accuracy") or 0.0)
    ml_total = float(summary.get("ml_accuracy") or 0.0)
    delta = ml_total - baseline_total

    baseline_dots = "\n".join(
        f'<circle class="baseline-dot" cx="{x:.0f}" cy="{_y(v):.2f}" r="8"/>'
        for x, v in zip(xs, baseline)
    )
    ml_dots = "\n".join(
        f'<circle class="ml-dot" cx="{x:.0f}" cy="{_y(v):.2f}" r="8"/>'
        for x, v in zip(xs, ml)
    )
    baseline_labels = "\n".join(
        f'<text class="value" x="{x:.0f}" y="{_y(v) + 24:.2f}" text-anchor="middle">{_pct(v)}</text>'
        for x, v in zip(xs, baseline)
    )
    ml_labels = "\n".join(
        f'<text class="value" x="{x:.0f}" y="{_y(v) - 18:.2f}" text-anchor="middle">{_pct(v)}</text>'
        for x, v in zip(xs, ml)
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="680" viewBox="0 0 1100 680">
  <defs>
    <style>
      .bg {{ fill: #f7f9fb; }}
      .panel {{ fill: #ffffff; stroke: #d7dee8; stroke-width: 1.5; }}
      .title {{ font: 700 34px Arial, sans-serif; fill: #142033; }}
      .subtitle {{ font: 400 18px Arial, sans-serif; fill: #5b6678; }}
      .axis {{ stroke: #1f2937; stroke-width: 2; }}
      .grid {{ stroke: #e6ebf1; stroke-width: 1; }}
      .tick {{ font: 14px Arial, sans-serif; fill: #526071; }}
      .label {{ font: 700 17px Arial, sans-serif; fill: #243244; }}
      .legend {{ font: 700 16px Arial, sans-serif; fill: #263447; }}
      .value {{ font: 700 15px Arial, sans-serif; fill: #263447; }}
      .note {{ font: 400 14px Arial, sans-serif; fill: #64748b; }}
      .baseline {{ fill: none; stroke: #6b7280; stroke-width: 4; }}
      .ml {{ fill: none; stroke: #00a99d; stroke-width: 4; }}
      .baseline-dot {{ fill: #6b7280; stroke: #ffffff; stroke-width: 3; }}
      .ml-dot {{ fill: #00a99d; stroke: #ffffff; stroke-width: 3; }}
    </style>
  </defs>
  <rect class="bg" width="1100" height="680"/>
  <rect class="panel" x="58" y="44" width="984" height="584" rx="18"/>
  <text class="title" x="92" y="96">{title}</text>
  <text class="subtitle" x="92" y="126">{subtitle}</text>
  <line class="grid" x1="150" y1="530" x2="970" y2="530"/>
  <line class="grid" x1="150" y1="458" x2="970" y2="458"/>
  <line class="grid" x1="150" y1="386" x2="970" y2="386"/>
  <line class="grid" x1="150" y1="314" x2="970" y2="314"/>
  <line class="grid" x1="150" y1="242" x2="970" y2="242"/>
  <line class="grid" x1="150" y1="170" x2="970" y2="170"/>
  <text class="tick" x="104" y="535">0%</text>
  <text class="tick" x="96" y="463">20%</text>
  <text class="tick" x="96" y="391">40%</text>
  <text class="tick" x="96" y="319">60%</text>
  <text class="tick" x="96" y="247">80%</text>
  <text class="tick" x="88" y="175">100%</text>
  <line class="axis" x1="150" y1="170" x2="150" y2="530"/>
  <line class="axis" x1="150" y1="530" x2="970" y2="530"/>
  <text class="label" x="256" y="570" text-anchor="middle">Low</text>
  <text class="label" x="560" y="570" text-anchor="middle">Medium</text>
  <text class="label" x="864" y="570" text-anchor="middle">High</text>
  <text class="note" x="560" y="600" text-anchor="middle">Entropy regime from normalized candidate-score entropy</text>
  <text class="label" x="560" y="646" text-anchor="middle">Entropy regime</text>
  <text class="label" transform="translate(42,350) rotate(-90)" text-anchor="middle">Top-1 accuracy</text>
  <polyline class="baseline" points="{baseline_points}"/>
  <polyline class="ml" points="{ml_points}"/>
  {baseline_dots}
  {ml_dots}
  {baseline_labels}
  {ml_labels}
  <line x1="740" y1="92" x2="790" y2="92" class="baseline"/>
  <circle class="baseline-dot" cx="765" cy="92" r="6"/>
  <text class="legend" x="802" y="98">Baseline</text>
  <line x1="740" y1="122" x2="790" y2="122" class="ml"/>
  <circle class="ml-dot" cx="765" cy="122" r="6"/>
  <text class="legend" x="802" y="128">ML reranker</text>
  <rect x="690" y="485" width="280" height="72" rx="10" fill="#e8f7f4" stroke="#b9e7df"/>
  <text class="legend" x="710" y="512">Overall improvement</text>
  <text class="value" x="710" y="538">{_pct(baseline_total)} → {_pct(ml_total)}  ({delta * 100:+.1f} pts)</text>
</svg>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot entropy-regime baseline vs ML accuracy as SVG.")
    parser.add_argument("--report", required=True, help="JSON from compare_entropy_regime_selection.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="Accuracy by Entropy Regime")
    parser.add_argument(
        "--subtitle",
        default="Baseline selection vs ML reranking on the True Demand KGQA test set",
    )
    args = parser.parse_args()
    report = _load_report(args.report)
    svg = build_svg(report, args.title, args.subtitle)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(svg, encoding="utf-8")
    print(f"SVG: {args.out}")


if __name__ == "__main__":
    main()
