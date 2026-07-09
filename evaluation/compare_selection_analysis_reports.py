#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary_row(name: str, path: str) -> Dict[str, object]:
    payload = _load_json(path)
    summary = dict(payload.get("summary") or {})
    total = int(summary.get("total") or 0)
    top1 = int(summary.get("top1_correct") or 0)
    any_correct = int(summary.get("any_correct") or 0)
    return {
        "name": name,
        "path": path,
        "total": total,
        "top1_correct": top1,
        "top1_rate": top1 / total if total else 0.0,
        "any_correct": any_correct,
        "any_rate": any_correct / total if total else 0.0,
        "ranking_failures": int(summary.get("ranking_failures") or summary.get("selection_failures") or max(any_correct - top1, 0)),
        "generation_failures": int(summary.get("generation_failures") or max(total - any_correct, 0)),
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _write_md(path: str, rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Selection Analysis Comparison",
        "",
        "| Mode | Questions | Top-1 | Any Correct | Ranking Failures | Generation Failures |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['total']} | "
            f"{row['top1_correct']} ({_fmt_pct(float(row['top1_rate']))}) | "
            f"{row['any_correct']} ({_fmt_pct(float(row['any_rate']))}) | "
            f"{row['ranking_failures']} | {row['generation_failures']} |"
        )
    lines.append("")
    if len(rows) >= 2:
        base = rows[0]
        for row in rows[1:]:
            delta = float(row["top1_rate"]) - float(base["top1_rate"])
            lines.append(
                f"- {row['name']} vs {base['name']}: Top-1 delta {_fmt_pct(delta)} "
                f"({int(row['top1_correct']) - int(base['top1_correct']):+d} questions)."
            )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare selection analysis JSON summaries.")
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="NAME=PATH. Repeat for each report, first report is baseline for deltas.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for item in args.report:
        if "=" not in item:
            raise RuntimeError("--report must use NAME=PATH")
        name, path = item.split("=", 1)
        rows.append(_summary_row(name.strip(), path.strip()))

    payload = {"reports": rows}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(args.out_md, rows)

    print("===== SELECTION ANALYSIS COMPARISON =====")
    for row in rows:
        print(
            f"{row['name']}: top1={row['top1_correct']}/{row['total']} "
            f"({row['top1_rate']:.3f}), any={row['any_correct']}/{row['total']} "
            f"({row['any_rate']:.3f})"
        )
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
