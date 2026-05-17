#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BENCHMARK_VARIANT_RE = re.compile(r"^\[Infineon benchmark variant \d+\]\s*", re.I)

from evaluation.audit_generated_benchmark_wording import row_warnings


class TextClient(Protocol):
    def generate_text(self, prompt: str) -> str:
        ...


def _clean_question(question: str) -> str:
    return BENCHMARK_VARIANT_RE.sub("", str(question or "").strip())


def _prompt(row: Dict[str, object], previous_questions: Optional[List[str]] = None) -> str:
    previous = "\n".join(f"- {question}" for question in (previous_questions or [])) or "- none"
    return f"""Rewrite one graph-question for a benchmark.

Return ONLY JSON in this exact form:
{{"question": "..."}}

Goal:
- Keep the intended graph answer compatible with the original gold query.
- Write one natural question an Infineon business user could ask.
- Do not mention SPARQL, RDF, graph internals, or "Infineon benchmark".
- Do not add facts that are not present in the source question.
- Keep the same business topic and requested dimensions.
- Preserve the original measure nouns when they matter:
  - percentage / percentage change must stay percentage / percentage change.
  - Do not replace percentage with share.
  - participant count / participants must stay participant-based, not become inventory quantity or generic records.
  - inventory trend must stay trend, not become time / over time unless time is present in the source.
  - yearly / monthly / quarterly wording must not be added or removed unless already implied by the source.
- Preserve the core answer intent of the gold query:
  - ranking_top must still ask for highest / largest / top / maximum.
  - count must still ask how many / count / number.
  - average must still ask for average / mean, not typical.
  - sum must still ask for total / sum.
- raw_or_lookup must still ask for raw values / list / lookup values.

Target ambiguity:
- low: explicit about aggregation / requested result shape.
- mid: realistic business wording; retain the core answer shape but allow some business shorthand.
- high: genuinely plausible business wording; preserve the core answer shape, but allow ambiguity in secondary dimensions, filters, or phrasing when natural.

Family: {row["family"]}
Answer shape intended by the gold query: {row["answer_shape"]}
Target ambiguity label: {row["target_ambiguity_label"]}
Source question: {_clean_question(str(row["example_question"]))}
Gold-query measure clue: {row["answer_shape"]}
Avoid exact reuse of these previous rewrites for this same template:
{previous}
"""


def _parse_question(text: str) -> str:
    cleaned = text.strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise ValueError(f"Could not parse JSON response: {cleaned[:300]!r}")
        parsed = json.loads(match.group(0))
    question = str(parsed.get("question") or "").strip()
    if not question:
        raise ValueError(f"Missing question in response: {cleaned[:300]!r}")
    return question.rstrip(".") + ("?" if not question.endswith("?") else "")


def _normalize_measure_wording(question: str, answer_shape: str) -> str:
    if answer_shape == "average" and re.search(r"\btypical\b", question, flags=re.I):
        return re.sub(r"\btypical\b", "average", question, flags=re.I)
    return question


def generate_rows(
    plan: Dict[str, object],
    *,
    client: TextClient,
    limit: Optional[int] = None,
    existing_rows: Optional[List[Dict[str, object]]] = None,
    progress: bool = False,
    request_pause_sec: float = 0.0,
    on_row: Optional[Callable[[List[Dict[str, object]]], None]] = None,
) -> List[Dict[str, object]]:
    out = list(existing_rows or [])
    start = len(out)
    plan_rows = list(plan["rows"])
    stop = len(plan_rows) if limit is None else min(len(plan_rows), limit)
    by_template_questions: Dict[str, List[str]] = {}
    for existing in out:
        by_template_questions.setdefault(str(existing["template_id"]), []).append(str(existing["question"]))
    for idx in range(start, stop):
        row = plan_rows[idx]
        template_id = str(row["template_id"])
        previous_questions = by_template_questions.setdefault(template_id, [])
        question = ""
        warnings: List[str] = []
        source_question = _clean_question(str(row["example_question"]))
        for _attempt in range(3):
            if request_pause_sec:
                time.sleep(request_pause_sec)
            try:
                question = _parse_question(client.generate_text(_prompt(row, previous_questions)))
            except ValueError:
                continue
            question = _normalize_measure_wording(question, str(row["answer_shape"]))
            candidate = {
                "question": question,
                "answer_shape": row["answer_shape"],
                "source_question": source_question,
            }
            warnings = row_warnings(candidate)
            duplicate = question.strip().lower() in {item.strip().lower() for item in previous_questions}
            if not duplicate and not warnings:
                break
        if not question:
            raise ValueError(f"Could not generate a parseable rewrite for plan row {idx + 1}.")
        previous_questions.append(question)
        out.append(
            {
                "id": f"FINALKGQA{idx + 1:03d}",
                "question": question,
                "query": row["query"],
                "topic": row["family"],
                "family": row["family"],
                "answer_shape": row["answer_shape"],
                "ambiguity_label": row["target_ambiguity_label"],
                "template_id": row["template_id"],
                "seed_id": row.get("source_id"),
                "seed_ambiguity_label": row.get("seed_ambiguity_label"),
                "source_question": source_question,
                "wording_warnings": warnings,
            }
        )
        if on_row is not None:
            on_row(out)
        if progress:
            print(f"[{idx + 1}/{stop}] {out[-1]['id']} - {question}", flush=True)
    return out


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "total": len(rows),
        "families": dict(Counter(str(row["family"]) for row in rows)),
        "answer_shapes": dict(Counter(str(row["answer_shape"]) for row in rows)),
        "ambiguity": dict(Counter(str(row["ambiguity_label"]) for row in rows)),
        "unique_templates": len({str(row["template_id"]) for row in rows}),
        "unique_questions": len({str(row["question"]).strip().lower() for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a natural-language KGQA benchmark using controlled LLM rewrites.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--request-pause-sec",
        type=float,
        default=float(os.environ.get("BENCHMARK_REWRITE_PAUSE_SEC", "2.1")),
        help="Pause before each LLM rewrite request to respect gateway rate limits.",
    )
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    existing_rows: List[Dict[str, object]] = []
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        existing_rows = json.loads(out_path.read_text(encoding="utf-8"))

    from llm.client import InfineonGPTClient

    client = InfineonGPTClient(temperature=0.2, max_tokens=300)

    def _save_progress(current_rows: List[Dict[str, object]]) -> None:
        out_path.write_text(json.dumps(current_rows, indent=2) + "\n", encoding="utf-8")

    rows = generate_rows(
        plan,
        client=client,
        limit=(args.limit or None),
        existing_rows=existing_rows,
        progress=args.progress,
        request_pause_sec=args.request_pause_sec,
        on_row=_save_progress,
    )
    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    summary = summarize(rows)
    print("===== FINAL KGQA BENCHMARK LLM DRAFT =====")
    print(f"Total: {summary['total']}")
    print(f"Unique templates: {summary['unique_templates']}")
    print(f"Unique questions: {summary['unique_questions']}")
    print(f"Ambiguity: {summary['ambiguity']}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
