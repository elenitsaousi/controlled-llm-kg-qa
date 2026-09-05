"""Build a deterministic Digital Reference ontology definition benchmark.

The generated benchmark is intentionally answerable without the LLM. Each row
asks for the definition of one ontology term and stores the expected ontology
term metadata. The evaluation script then checks whether the runtime router
returns the same DR term and a non-empty ontology-grounded answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg.dr_ontology import DEFAULT_DR_ONTOLOGY_PATH, DROntologyTerm, _load_dr_terms


PRIORITY_TERMS = (
    "demand",
    "forecast",
    "product",
    "supply chain",
    "customer",
    "supplier",
    "order",
    "sales",
    "capacity",
    "inventory",
    "market",
    "organization",
    "process",
    "component",
    "material",
)

QUESTION_TEMPLATES = (
    "What is {label}?",
    "Define {label}.",
    "What does {label} mean?",
)


def _safe_label(label: str) -> bool:
    text = str(label or "").strip()
    if len(text) < 3 or len(text) > 70:
        return False
    if any(ch in text for ch in "{}[]<>|"):
        return False
    return any(ch.isalpha() for ch in text)


def _unique_terms(terms_by_alias: Dict[str, DROntologyTerm]) -> List[DROntologyTerm]:
    by_uri: Dict[str, DROntologyTerm] = {}
    for term in terms_by_alias.values():
        if term.uri not in by_uri and term.definition and _safe_label(term.label):
            by_uri[term.uri] = term
    terms = list(by_uri.values())

    def key(term: DROntologyTerm) -> tuple[int, str]:
        label = term.label.lower()
        priority = 0 if any(seed in label for seed in PRIORITY_TERMS) else 1
        return (priority, label)

    return sorted(terms, key=key)


def build_benchmark(dr_ontology: str, *, limit: int) -> List[Dict[str, str]]:
    terms = _unique_terms(_load_dr_terms(str(Path(dr_ontology).expanduser())))
    rows: List[Dict[str, str]] = []
    for idx, term in enumerate(terms[:limit], start=1):
        template = QUESTION_TEMPLATES[(idx - 1) % len(QUESTION_TEMPLATES)]
        rows.append(
            {
                "id": f"DRKGQA{idx:04d}",
                "question": template.format(label=term.label),
                "expected_route": "definition",
                "expected_source": "digital_reference_ontology",
                "expected_term": term.label,
                "expected_kind": term.kind,
                "expected_uri": term.uri,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DR ontology definition benchmark JSON file.")
    parser.add_argument(
        "--dr-ontology",
        default=os.getenv("TRUE_DEMAND_DR_ONTOLOGY_PATH") or str(DEFAULT_DR_ONTOLOGY_PATH),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", default="evaluation/question_sets/dr_ontology_benchmark.json")
    args = parser.parse_args()

    rows = build_benchmark(args.dr_ontology, limit=args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("===== DR ONTOLOGY BENCHMARK BUILDER =====")
    print(f"DR ontology: {args.dr_ontology}")
    print(f"Rows: {len(rows)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
