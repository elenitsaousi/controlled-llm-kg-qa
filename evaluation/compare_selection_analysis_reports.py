#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


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


def _execution_plausible(candidate: Dict[str, object]) -> bool:
    if not candidate:
        return False
    if candidate.get("execution_error"):
        return False
    if bool(candidate.get("execution_has_rows")):
        return True
    try:
        return int(candidate.get("execution_row_count") or 0) > 0
    except Exception:
        return False


def _selection_execution_row(path: str) -> Dict[str, object]:
    payload = _load_json(path)
    details = list(payload.get("details") or [])
    total = 0
    plausible = 0
    no_candidate = 0
    execution_errors = 0
    empty_selected = 0
    for detail in details:
        if not isinstance(detail, dict):
            continue
        total += 1
        candidates = list(detail.get("candidates") or [])
        if not candidates:
            no_candidate += 1
            continue
        selected = candidates[0]
        if selected.get("execution_error"):
            execution_errors += 1
        elif not _execution_plausible(selected):
            empty_selected += 1
        plausible += int(_execution_plausible(selected))
    return {
        "selection_path": path,
        "execution_plausible": plausible,
        "execution_plausible_rate": plausible / total if total else 0.0,
        "selected_no_candidate": no_candidate,
        "selected_execution_errors": execution_errors,
        "selected_empty": empty_selected,
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _write_md(path: str, rows: List[Dict[str, object]]) -> None:
    include_execution = any("execution_plausible" in row for row in rows)
    lines = [
        "# Selection Analysis Comparison",
        "",
    ]
    if include_execution:
        lines.extend(
            [
                "| Mode | Questions | Top-1 | Any Correct | Execution-plausible selected query | Ranking Failures | Generation Failures |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| Mode | Questions | Top-1 | Any Correct | Ranking Failures | Generation Failures |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
    for row in rows:
        base = (
            f"| {row['name']} | {row['total']} | "
            f"{row['top1_correct']} ({_fmt_pct(float(row['top1_rate']))}) | "
            f"{row['any_correct']} ({_fmt_pct(float(row['any_rate']))}) | "
        )
        if include_execution:
            if "execution_plausible" in row:
                execution_cell = (
                    f"{row['execution_plausible']} "
                    f"({_fmt_pct(float(row['execution_plausible_rate']))})"
                )
            else:
                execution_cell = "n/a"
            lines.append(
                base
                + f"{execution_cell} | "
                + f"{row['ranking_failures']} | {row['generation_failures']} |"
            )
        else:
            lines.append(base + f"{row['ranking_failures']} | {row['generation_failures']} |")
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
    parser.add_argument(
        "--selection",
        action="append",
        default=[],
        help=(
            "Optional NAME=PATH to the corresponding selected-candidate results JSON. "
            "When provided, adds execution-plausible selected query metrics."
        ),
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    selections: Dict[str, Dict[str, object]] = {}
    for item in args.selection:
        if "=" not in item:
            raise RuntimeError("--selection must use NAME=PATH")
        name, path = item.split("=", 1)
        selections[name.strip()] = _selection_execution_row(path.strip())

    rows: List[Dict[str, object]] = []
    for item in args.report:
        if "=" not in item:
            raise RuntimeError("--report must use NAME=PATH")
        name, path = item.split("=", 1)
        row = _summary_row(name.strip(), path.strip())
        if row["name"] in selections:
            row.update(selections[str(row["name"])])
        rows.append(row)

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
        if "execution_plausible" in row:
            print(
                f"  execution-plausible={row['execution_plausible']}/{row['total']} "
                f"({row['execution_plausible_rate']:.3f})"
            )
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
