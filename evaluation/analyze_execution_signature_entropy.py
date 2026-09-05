#!/usr/bin/env python3
"""Compare raw candidate entropy with execution-signature clustered entropy.

This diagnostic addresses the limitation that distinct SPARQL strings can be
equivalent for the user if they return the same projected rows. When row
previews are present in a result file, candidates are clustered by a stable
execution signature before entropy is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _details(payload: Dict[str, object]) -> List[Dict[str, object]]:
    rows = payload.get("details") or payload.get("rows") or []
    return rows if isinstance(rows, list) else []


def _candidate_score(candidate: Dict[str, object], score_key: str, fallback_rank: int) -> float:
    for key in [score_key, "ml_score", "selection_score", "score"]:
        value = candidate.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return -float(fallback_rank)


def _softmax(scores: List[float], temperature: float) -> List[float]:
    if not scores:
        return []
    temp = max(float(temperature), 1e-9)
    shifted = [(s - max(scores)) / temp for s in scores]
    exps = [math.exp(max(min(x, 60), -60)) for x in shifted]
    total = sum(exps)
    return [x / total for x in exps] if total else [1.0 / len(scores)] * len(scores)


def _entropy(probs: List[float]) -> float:
    probs = [p for p in probs if p > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(probs))


def _canonical_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip().lower()


def _rows_from_candidate(candidate: Dict[str, object]) -> object:
    for key in ["row_preview", "preview_rows", "rows", "result_rows", "graph_rows"]:
        value = candidate.get(key)
        if value:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except Exception:
                    return value
            return value
    execution = candidate.get("execution") or candidate.get("result") or {}
    if isinstance(execution, dict):
        for key in ["row_preview", "preview_rows", "rows", "result_rows", "graph_rows"]:
            value = execution.get(key)
            if value:
                return value
    return None


def _signature(candidate: Dict[str, object]) -> Tuple[str, str]:
    rows = _rows_from_candidate(candidate)
    if rows is not None:
        payload = {"kind": "execution", "rows": rows}
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return "execution", hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    query = _canonical_query(str(candidate.get("query") or ""))
    return "query_text", hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


def analyze(results_path: str, score_key: str, temperature: float) -> Dict[str, object]:
    payload = _load_json(results_path)
    rows = []
    raw_values = []
    clustered_values = []
    clustered_questions = 0
    execution_signature_questions = 0

    for detail in _details(payload):
        if not isinstance(detail, dict):
            continue
        candidates = detail.get("candidates") or []
        if not isinstance(candidates, list) or len(candidates) < 2:
            continue
        scores = [_candidate_score(c, score_key, idx) for idx, c in enumerate(candidates)]
        probs = _softmax(scores, temperature)
        raw_h = _entropy(probs)

        grouped: Dict[str, float] = {}
        kinds = set()
        for prob, candidate in zip(probs, candidates):
            kind, sig = _signature(candidate)
            kinds.add(kind)
            grouped[sig] = grouped.get(sig, 0.0) + prob
        clustered_h = _entropy(list(grouped.values()))
        raw_values.append(raw_h)
        clustered_values.append(clustered_h)
        if len(grouped) < len(candidates):
            clustered_questions += 1
        if "execution" in kinds:
            execution_signature_questions += 1
        rows.append(
            {
                "id": detail.get("id") or detail.get("question_id") or detail.get("request_id"),
                "question": detail.get("question") or detail.get("effective_question"),
                "candidate_count": len(candidates),
                "signature_count": len(grouped),
                "raw_entropy": raw_h,
                "signature_entropy": clustered_h,
                "entropy_reduction": raw_h - clustered_h,
                "signature_kind": "execution" if "execution" in kinds else "query_text",
            }
        )

    rows.sort(key=lambda row: (-float(row["entropy_reduction"]), str(row["id"])))
    total = len(rows)
    avg_raw = sum(raw_values) / total if total else 0.0
    avg_sig = sum(clustered_values) / total if total else 0.0
    return {
        "results": results_path,
        "score_key": score_key,
        "temperature": temperature,
        "summary": {
            "questions": total,
            "avg_raw_entropy": avg_raw,
            "avg_signature_entropy": avg_sig,
            "avg_entropy_reduction": avg_raw - avg_sig,
            "questions_with_collapsed_signatures": clustered_questions,
            "questions_with_execution_signatures": execution_signature_questions,
        },
        "top_entropy_reductions": rows[:50],
        "interpretation": (
            "Signature clustering avoids counting equivalent result-producing queries as fully "
            "separate competitors. If execution previews are unavailable, the diagnostic falls "
            "back to query-text signatures and should be treated as a limitation analysis rather "
            "than final semantic-equivalence evidence."
        ),
    }


def _write_md(report: Dict[str, object], out_md: str) -> None:
    s = report["summary"]
    lines = [
        "# Execution-Signature Entropy Diagnostic",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Questions | {s['questions']} |",
        f"| Avg raw entropy | {s['avg_raw_entropy']:.3f} |",
        f"| Avg signature entropy | {s['avg_signature_entropy']:.3f} |",
        f"| Avg entropy reduction | {s['avg_entropy_reduction']:.3f} |",
        f"| Questions with collapsed signatures | {s['questions_with_collapsed_signatures']} |",
        f"| Questions with execution signatures | {s['questions_with_execution_signatures']} |",
        "",
        "## Interpretation",
        "",
        str(report["interpretation"]),
        "",
        "## Largest Reductions",
        "",
    ]
    for row in report["top_entropy_reductions"][:20]:
        lines.append(
            f"- `{row['id']}` candidates={row['candidate_count']} signatures={row['signature_count']} "
            f"raw={row['raw_entropy']:.3f} clustered={row['signature_entropy']:.3f}: {row.get('question')}"
        )
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute entropy before and after execution-signature clustering.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--score-key", default="ml_score")
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    report = analyze(args.results, args.score_key, args.temperature)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _write_md(report, args.out_md)
    print("===== EXECUTION-SIGNATURE ENTROPY =====")
    print(json.dumps(report["summary"], indent=2))
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
