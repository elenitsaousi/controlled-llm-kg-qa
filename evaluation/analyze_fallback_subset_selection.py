#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _candidate_label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "")).strip().lower()


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


def _dataset_indexes(dataset_path: str) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    rows = _load_json(dataset_path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Dataset must be a JSON list: {dataset_path}")
    by_id = {str(row.get("id", "")).strip(): row for row in rows if isinstance(row, dict)}
    by_question = {
        _norm(str(row.get("question", ""))): row
        for row in rows
        if isinstance(row, dict) and _norm(str(row.get("question", "")))
    }
    return by_id, by_question


def _load_fallback_rows(audit_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(audit_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = str(row.get("system_mode", "")).strip().lower()
            route = str(row.get("route", "")).strip().lower()
            if mode == "llm_ranking" or route == "llm_ranking":
                rows.append(row)
    return rows


def _map_fallback_rows(
    fallback_rows: List[Dict[str, str]],
    dataset_by_id: Dict[str, Dict[str, object]],
    dataset_by_question: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    mapped: List[Dict[str, object]] = []
    unmatched: List[Dict[str, str]] = []
    seen_ids = set()
    for row in fallback_rows:
        gold_id = str(row.get("source_gold_id") or "").strip()
        dataset_row = dataset_by_id.get(gold_id) if gold_id else None
        if dataset_row is None:
            dataset_row = dataset_by_question.get(_norm(row.get("question", "")))
        if dataset_row is None:
            unmatched.append(row)
            continue
        qid = str(dataset_row.get("id", "")).strip()
        if qid in seen_ids:
            continue
        seen_ids.add(qid)
        mapped.append(
            {
                "audit_request_id": row.get("request_id"),
                "audit_question": row.get("question"),
                "audit_correctness": str(row.get("correctness", "")).strip().lower(),
                "audit_selected_source": row.get("selected_source"),
                "audit_selected_query": row.get("selected_query"),
                "dataset_id": qid,
                "dataset_question": dataset_row.get("question"),
                "family": dataset_row.get("family") or dataset_row.get("topic") or "unknown",
            }
        )
    return mapped, unmatched


def _details_by_id(selection_path: str) -> Dict[str, Dict[str, object]]:
    payload = _load_json(selection_path)
    return {
        str(detail.get("id", "")).strip(): detail
        for detail in list(payload.get("details") or [])
        if isinstance(detail, dict) and str(detail.get("id", "")).strip()
    }


def _mode_row(name: str, path: str, mapped_rows: List[Dict[str, object]]) -> Dict[str, object]:
    details = _details_by_id(path)
    total = 0
    top1 = 0
    any_correct = 0
    execution_plausible = 0
    missing_details = 0
    selected_empty = 0
    selected_errors = 0
    family_counts: Dict[str, Counter] = defaultdict(Counter)

    for mapped in mapped_rows:
        qid = str(mapped["dataset_id"])
        family = str(mapped.get("family") or "unknown")
        detail = details.get(qid)
        if detail is None:
            missing_details += 1
            continue
        total += 1
        candidates = list(detail.get("candidates") or [])
        selected = candidates[0] if candidates else {}
        strict = bool(detail.get("top1_correct")) or _candidate_label(selected) == "correct"
        any_ok = bool(detail.get("any_correct")) or any(_candidate_label(c) == "correct" for c in candidates)
        plausible = _execution_plausible(selected)
        top1 += int(strict)
        any_correct += int(any_ok)
        execution_plausible += int(plausible)
        if selected.get("execution_error"):
            selected_errors += 1
        elif not plausible:
            selected_empty += 1

        family_counts[family]["total"] += 1
        family_counts[family]["top1"] += int(strict)
        family_counts[family]["any_correct"] += int(any_ok)
        family_counts[family]["execution_plausible"] += int(plausible)

    family_rows = []
    for family, counts in sorted(family_counts.items()):
        fam_total = counts["total"] or 1
        family_rows.append(
            {
                "family": family,
                "total": counts["total"],
                "top1_correct": counts["top1"],
                "top1_rate": counts["top1"] / fam_total,
                "any_correct": counts["any_correct"],
                "any_rate": counts["any_correct"] / fam_total,
                "execution_plausible": counts["execution_plausible"],
                "execution_plausible_rate": counts["execution_plausible"] / fam_total,
            }
        )

    denom = total or 1
    return {
        "name": name,
        "selection_path": path,
        "questions": total,
        "missing_details": missing_details,
        "top1_correct": top1,
        "top1_rate": top1 / denom,
        "any_correct": any_correct,
        "any_rate": any_correct / denom,
        "execution_plausible": execution_plausible,
        "execution_plausible_rate": execution_plausible / denom,
        "ranking_failures": max(any_correct - top1, 0),
        "generation_failures": max(total - any_correct, 0),
        "selected_empty": selected_empty,
        "selected_errors": selected_errors,
        "by_family": family_rows,
    }


def analyze(
    *,
    audit_csv: str,
    dataset_path: str,
    selections: List[Tuple[str, str]],
) -> Dict[str, object]:
    dataset_by_id, dataset_by_question = _dataset_indexes(dataset_path)
    fallback_rows = _load_fallback_rows(audit_csv)
    mapped_rows, unmatched_rows = _map_fallback_rows(fallback_rows, dataset_by_id, dataset_by_question)

    final_correct = sum(1 for row in fallback_rows if str(row.get("correctness", "")).strip().lower() == "correct")
    mapped_final_correct = sum(1 for row in mapped_rows if str(row.get("audit_correctness", "")).strip().lower() == "correct")
    mode_rows = [_mode_row(name, path, mapped_rows) for name, path in selections]

    return {
        "audit_csv": audit_csv,
        "dataset": dataset_path,
        "fallback_rows": len(fallback_rows),
        "mapped_fallback_rows": len(mapped_rows),
        "unmatched_fallback_rows": len(unmatched_rows),
        "final_answer_level": {
            "correct": final_correct,
            "total": len(fallback_rows),
            "accuracy": final_correct / len(fallback_rows) if fallback_rows else 0.0,
            "mapped_correct": mapped_final_correct,
            "mapped_total": len(mapped_rows),
            "mapped_accuracy": mapped_final_correct / len(mapped_rows) if mapped_rows else 0.0,
        },
        "modes": mode_rows,
        "mapped_rows": mapped_rows,
        "unmatched_rows": [
            {
                "request_id": row.get("request_id"),
                "question": row.get("question"),
                "topic": row.get("topic"),
                "correctness": row.get("correctness"),
                "selected_source": row.get("selected_source"),
            }
            for row in unmatched_rows
        ],
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _write_md(path: str, report: Dict[str, object]) -> None:
    answer = report["final_answer_level"]
    lines = [
        "# LLM Fallback Subset Selection Analysis",
        "",
        f"Audit CSV: `{report['audit_csv']}`",
        f"Dataset: `{report['dataset']}`",
        "",
        "## Mapping",
        "",
        f"- Final-system LLM fallback rows: {report['fallback_rows']}",
        f"- Mapped to repaired KG candidate IDs: {report['mapped_fallback_rows']}",
        f"- Unmatched fallback rows: {report['unmatched_fallback_rows']}",
        "",
        "## Final Answer-Level Fallback Accuracy",
        "",
        f"- All fallback rows: {answer['correct']}/{answer['total']} ({_fmt_pct(answer['accuracy'])})",
        f"- Mapped fallback rows only: {answer['mapped_correct']}/{answer['mapped_total']} ({_fmt_pct(answer['mapped_accuracy'])})",
        "",
        "## Candidate Selection on Mapped Fallback Subset",
        "",
        "| Mode | Questions | Top-1 | Any Correct | Exec.-plausible | Rank fail. | Gen. fail. | Empty selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["modes"]:
        lines.append(
            f"| {row['name']} | {row['questions']} | "
            f"{row['top1_correct']} ({_fmt_pct(row['top1_rate'])}) | "
            f"{row['any_correct']} ({_fmt_pct(row['any_rate'])}) | "
            f"{row['execution_plausible']} ({_fmt_pct(row['execution_plausible_rate'])}) | "
            f"{row['ranking_failures']} | {row['generation_failures']} | {row['selected_empty']} |"
        )

    if report["unmatched_rows"]:
        lines.extend(["", "## Unmatched Fallback Rows", ""])
        for row in report["unmatched_rows"]:
            lines.append(f"- `{row['request_id']}`: {row['question']}")

    lines.extend(
        [
            "",
            "Interpretation: this table evaluates only the questions that the final system actually routed to the LLM fallback layer. ",
            "The answer-level fallback accuracy comes from the manual/full-system audit. ",
            "The Top-1, Any-Correct, and execution-plausible columns are candidate-selection diagnostics computed after mapping those fallback questions back to the repaired KG candidate benchmark.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate raw/schema/ML selection only on final-system LLM fallback questions."
    )
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--selection",
        action="append",
        required=True,
        help="NAME=PATH to a selected-candidate results JSON. Repeat for raw/schema/ML modes.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    selections = []
    for item in args.selection:
        if "=" not in item:
            raise RuntimeError("--selection must use NAME=PATH")
        name, path = item.split("=", 1)
        selections.append((name.strip(), path.strip()))

    report = analyze(audit_csv=args.audit_csv, dataset_path=args.dataset, selections=selections)
    _write_json(args.out_json, report)
    _write_md(args.out_md, report)

    print("===== LLM FALLBACK SUBSET SELECTION ANALYSIS =====")
    print(f"Fallback rows: {report['fallback_rows']}")
    print(f"Mapped rows: {report['mapped_fallback_rows']}")
    print(f"Unmatched rows: {report['unmatched_fallback_rows']}")
    answer = report["final_answer_level"]
    print(f"Answer-level fallback: {answer['correct']}/{answer['total']} ({answer['accuracy']:.3f})")
    for row in report["modes"]:
        print(
            f"{row['name']}: top1={row['top1_correct']}/{row['questions']} "
            f"({row['top1_rate']:.3f}), any={row['any_correct']}/{row['questions']} "
            f"({row['any_rate']:.3f}), exec={row['execution_plausible']}/{row['questions']} "
            f"({row['execution_plausible_rate']:.3f})"
        )
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
