#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


VAR_RE = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
STRING_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _query_signature(query: str) -> str:
    text = " ".join(str(query or "").split())
    text = STRING_RE.sub("STR", text)
    text = NUMBER_RE.sub("NUM", text)
    text = VAR_RE.sub("?VAR", text)
    return "tmpl_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _family(row: Dict[str, object]) -> str:
    q = f"{row.get('question', '')} {row.get('query', '')}".lower()
    if "ordercancellation" in q or "order cancellation" in q or "cancellation" in q:
        return "order_cancellation"
    if "inventory" in q:
        return "inventory"
    if "shortage" in q:
        return "shortage"
    if "autonomous" in q or "sae" in q:
        return "autonomous_driving"
    if "vehiclesalesobservation" in q or "vehicle sales" in q or "yearly sales" in q:
        return "vehicle_sales"
    if "futuredemand" in q or "future demand" in q:
        return "future_demand"
    if "baseline" in q or "bl1" in q or "bl2" in q:
        return "current_demand_baselines"
    if "demandforregion" in q or "regional demand" in q or " by region" in q:
        return "regional_demand"
    return "catalog_lookup"


def _shape(query: str) -> str:
    q = str(query or "").upper()
    if "ORDER BY DESC" in q and "LIMIT 1" in q:
        return "ranking_top"
    if "COUNT(" in q:
        return "count"
    if "MAX(" in q:
        return "ranking_top"
    if "AVG(" in q:
        return "average"
    if "SUM(" in q:
        return "sum"
    return "raw_or_lookup"


def build_seed_bank(paths: Iterable[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen = set()
    for path in paths:
        for row in json.loads(Path(path).read_text(encoding="utf-8")):
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            signature = _query_signature(query)
            if signature in seen:
                continue
            seen.add(signature)
            rows.append(
                {
                    "template_id": signature,
                    "family": _family(row),
                    "answer_shape": _shape(query),
                    "ambiguity_label": row.get("ambiguity_label"),
                    "source_dataset": path,
                    "source_id": row.get("id"),
                    "example_question": row.get("question"),
                    "query": query,
                }
            )
    rows.sort(key=lambda row: (str(row["family"]), str(row["template_id"])))
    return rows


def _print_summary(rows: List[Dict[str, object]]) -> None:
    print("===== KGQA SEED BANK =====")
    print(f"Unique templates: {len(rows)}")
    print("Families:")
    for key, value in sorted(Counter(str(row["family"]) for row in rows).items()):
        print(f"  {key}: {value}")
    print("Answer shapes:")
    for key, value in sorted(Counter(str(row["answer_shape"]) for row in rows).items()):
        print(f"  {key}: {value}")


def summarize_seed_bank(rows: List[Dict[str, object]]) -> Dict[str, object]:
    families = sorted({str(row["family"]) for row in rows})
    shapes = sorted({str(row["answer_shape"]) for row in rows})
    matrix = {
        family: {
            shape: sum(
                1
                for row in rows
                if row["family"] == family and row["answer_shape"] == shape
            )
            for shape in shapes
        }
        for family in families
    }
    return {
        "unique_templates": len(rows),
        "families": dict(Counter(str(row["family"]) for row in rows)),
        "answer_shapes": dict(Counter(str(row["answer_shape"]) for row in rows)),
        "family_shape_matrix": matrix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized KGQA seed-template bank.")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = build_seed_bank(args.datasets)
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    _print_summary(rows)
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
