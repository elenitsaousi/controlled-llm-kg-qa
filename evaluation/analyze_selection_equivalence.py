#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _norm_query(query: str) -> str:
    return " ".join(str(query or "").split()).lower()


def _norm_question(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _request_id_to_final_id(request_id: str) -> str:
    digits = "".join(ch for ch in str(request_id or "") if ch.isdigit())
    if not digits:
        return ""
    return f"FINALKGQA{int(digits):03d}"


def _load_audit(path: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not path:
        return out
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = _request_id_to_final_id(row.get("request_id", ""))
            query = _norm_query(row.get("selected_query", ""))
            if qid and query:
                out[(qid, query)] = row
            question_key = _norm_question(row.get("question", ""))
            if question_key and query:
                out[(question_key, query)] = row
    return out


def _load_dataset(path: str) -> Dict[str, Dict[str, object]]:
    if not path:
        return {}
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise RuntimeError(f"Dataset must be a JSON list: {path}")
    return {str(row.get("id", "")).strip(): row for row in rows if isinstance(row, dict)}


def _selected_candidate(detail: Dict[str, object]) -> Dict[str, object]:
    candidates = list(detail.get("candidates") or [])
    return candidates[0] if candidates else {}


def _query_shape(query: str) -> str:
    q = str(query or "").lower()
    if "group by" in q:
        return "grouped"
    if any(token in q for token in ["count(", "sum(", "avg(", "min(", "max("]):
        return "aggregate_scalar"
    return "raw_or_lookup"


def _execution_plausible(candidate: Dict[str, object]) -> bool:
    if not candidate:
        return False
    if candidate.get("execution_error"):
        return False
    if bool(candidate.get("execution_has_rows")):
        return True
    row_count = candidate.get("execution_row_count")
    try:
        return int(row_count) > 0
    except Exception:
        return False


def analyze(
    *,
    results_path: str,
    dataset_path: str = "",
    audit_csv: str = "",
) -> Dict[str, object]:
    payload = _load_json(results_path)
    details = list(payload.get("details") or [])
    dataset = _load_dataset(dataset_path)
    audit = _load_audit(audit_csv)

    rows: List[Dict[str, object]] = []
    counts = Counter()
    family_counts: Dict[str, Counter] = {}
    for detail in details:
        qid = str(detail.get("id", "")).strip()
        question = str(detail.get("effective_question") or detail.get("question") or "")
        dataset_row = dataset.get(qid, {})
        family = (
            str(
                detail.get("family")
                or detail.get("topic")
                or dataset_row.get("family")
                or dataset_row.get("topic")
                or dataset_row.get("family_id")
                or "unknown"
            ).strip()
            or "unknown"
        )
        selected = _selected_candidate(detail)
        selected_query = str(selected.get("query", "") or "")
        selected_label = str(selected.get("label", "") or "").lower()
        strict_correct = selected_label == "correct"
        any_correct = bool(detail.get("any_correct"))
        plausible = _execution_plausible(selected)

        audit_row = audit.get((qid, _norm_query(selected_query)))
        if audit_row is None:
            audit_row = audit.get((_norm_question(question), _norm_query(selected_query)))
        audit_correctness = str((audit_row or {}).get("correctness", "")).strip().lower()
        answer_correct = audit_correctness == "correct"
        answer_incorrect = audit_correctness == "incorrect"
        answer_unclear = audit_correctness in {"unclear", "probably correct", "probably_correct"}
        audit_matched = audit_row is not None

        counts["total"] += 1
        counts["strict_correct"] += int(strict_correct)
        counts["any_correct"] += int(any_correct)
        counts["execution_plausible"] += int(plausible)
        counts["audit_matched"] += int(audit_matched)
        counts["answer_correct"] += int(answer_correct)
        counts["answer_incorrect"] += int(answer_incorrect)
        counts["answer_unclear"] += int(answer_unclear)
        if not strict_correct and answer_correct:
            counts["strict_wrong_answer_correct"] += 1
        if strict_correct and answer_incorrect:
            counts["strict_correct_answer_wrong"] += 1

        fam_counter = family_counts.setdefault(family, Counter())
        fam_counter["total"] += 1
        fam_counter["strict_correct"] += int(strict_correct)
        fam_counter["any_correct"] += int(any_correct)
        fam_counter["execution_plausible"] += int(plausible)
        fam_counter["audit_matched"] += int(audit_matched)
        fam_counter["answer_correct"] += int(answer_correct)

        rows.append(
            {
                "id": qid,
                "question": question,
                "family": family,
                "selected_label": selected_label,
                "strict_correct": strict_correct,
                "any_correct": any_correct,
                "execution_plausible": plausible,
                "shape": _query_shape(selected_query),
                "execution_row_count": selected.get("execution_row_count"),
                "execution_error": selected.get("execution_error"),
                "audit_matched": audit_matched,
                "answer_correctness": audit_correctness or None,
                "strict_wrong_answer_correct": bool((not strict_correct) and answer_correct),
                "selected_query": selected_query,
            }
        )

    total = counts["total"] or 1
    audit_total = counts["audit_matched"] or 1
    family_rows = []
    for family, counter in sorted(family_counts.items()):
        fam_total = counter["total"] or 1
        fam_audit_total = counter["audit_matched"] or 1
        family_rows.append(
            {
                "family": family,
                "total": counter["total"],
                "strict_correct": counter["strict_correct"],
                "strict_rate": counter["strict_correct"] / fam_total,
                "any_correct": counter["any_correct"],
                "any_rate": counter["any_correct"] / fam_total,
                "execution_plausible": counter["execution_plausible"],
                "execution_plausible_rate": counter["execution_plausible"] / fam_total,
                "audit_matched": counter["audit_matched"],
                "answer_correct": counter["answer_correct"],
                "answer_rate_on_matched": counter["answer_correct"] / fam_audit_total,
            }
        )

    return {
        "results": results_path,
        "dataset": dataset_path or None,
        "audit_csv": audit_csv or None,
        "summary": {
            "total": counts["total"],
            "strict_correct": counts["strict_correct"],
            "strict_rate": counts["strict_correct"] / total,
            "any_correct": counts["any_correct"],
            "any_rate": counts["any_correct"] / total,
            "execution_plausible": counts["execution_plausible"],
            "execution_plausible_rate": counts["execution_plausible"] / total,
            "audit_matched": counts["audit_matched"],
            "answer_correct": counts["answer_correct"],
            "answer_rate_on_matched": counts["answer_correct"] / audit_total,
            "answer_incorrect": counts["answer_incorrect"],
            "answer_unclear": counts["answer_unclear"],
            "strict_wrong_answer_correct": counts["strict_wrong_answer_correct"],
            "strict_correct_answer_wrong": counts["strict_correct_answer_wrong"],
        },
        "by_family": family_rows,
        "rows": rows,
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _write_md(path: str, report: Dict[str, object]) -> None:
    summary = report["summary"]
    lines = [
        "# Selection Equivalence Analysis",
        "",
        f"Results: `{report['results']}`",
        f"Dataset: `{report['dataset'] or 'not provided'}`",
        f"Audit CSV: `{report['audit_csv'] or 'not provided'}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Strict selected-query accuracy | {summary['strict_correct']}/{summary['total']} ({_fmt_pct(summary['strict_rate'])}) |",
        f"| Any-correct candidate coverage | {summary['any_correct']}/{summary['total']} ({_fmt_pct(summary['any_rate'])}) |",
        f"| Execution-plausible selected query | {summary['execution_plausible']}/{summary['total']} ({_fmt_pct(summary['execution_plausible_rate'])}) |",
        f"| Audit-matched selected answers | {summary['audit_matched']}/{summary['total']} |",
        f"| Answer-level accuracy on matched audit rows | {summary['answer_correct']}/{summary['audit_matched']} ({_fmt_pct(summary['answer_rate_on_matched'])}) |",
        f"| Strict-wrong but answer-correct | {summary['strict_wrong_answer_correct']} |",
        f"| Strict-correct but answer-wrong | {summary['strict_correct_answer_wrong']} |",
        "",
        "## By Family",
        "",
        "| Family | N | Strict | Any | Execution-plausible | Audit matched | Answer-level on matched |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_family"]:
        lines.append(
            f"| {row['family']} | {row['total']} | "
            f"{row['strict_correct']} ({_fmt_pct(row['strict_rate'])}) | "
            f"{row['any_correct']} ({_fmt_pct(row['any_rate'])}) | "
            f"{row['execution_plausible']} ({_fmt_pct(row['execution_plausible_rate'])}) | "
            f"{row['audit_matched']} | "
            f"{row['answer_correct']} ({_fmt_pct(row['answer_rate_on_matched'])}) |"
        )
    lines.extend(
        [
            "",
            "Interpretation: strict accuracy counts only the canonical gold candidate. ",
            "Answer-level accuracy counts whether the selected query produced an audited correct user-facing answer, when an audit row with the same selected query is available.",
        ]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare strict selected-query accuracy with execution/audit-equivalent accuracy."
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--audit-csv", default="")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(results_path=args.results, dataset_path=args.dataset, audit_csv=args.audit_csv)
    _write_json(args.out_json, report)
    _write_md(args.out_md, report)
    summary = report["summary"]
    print("===== SELECTION EQUIVALENCE ANALYSIS =====")
    print(f"Strict: {summary['strict_correct']}/{summary['total']} ({summary['strict_rate']:.3f})")
    print(f"Any: {summary['any_correct']}/{summary['total']} ({summary['any_rate']:.3f})")
    print(
        "Answer-level on matched audit rows: "
        f"{summary['answer_correct']}/{summary['audit_matched']} "
        f"({summary['answer_rate_on_matched']:.3f})"
    )
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
