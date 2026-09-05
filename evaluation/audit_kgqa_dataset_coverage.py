#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def _rows(path: str) -> List[Dict[str, object]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _shape(query: str) -> str:
    q = str(query or "").upper()
    if "COUNT(" in q:
        return "count"
    if "MAX(" in q or "ORDER BY DESC" in q and "LIMIT 1" in q:
        return "ranking_top"
    if "AVG(" in q:
        return "average"
    if "SUM(" in q:
        return "sum"
    return "raw_or_lookup"


def audit(paths: Iterable[str]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    by_dataset: Dict[str, int] = {}
    for path in paths:
        loaded = _rows(path)
        rows.extend(loaded)
        by_dataset[path] = len(loaded)

    return {
        "datasets": by_dataset,
        "total": len(rows),
        "topics": dict(Counter(str(row.get("topic") or "unknown") for row in rows)),
        "ambiguity": dict(Counter(str(row.get("ambiguity_label") or "unknown") for row in rows)),
        "answer_shapes": dict(Counter(_shape(str(row.get("query") or "")) for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit KGQA dataset coverage across splits.")
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    report = audit(args.datasets)
    print("===== KGQA DATASET COVERAGE AUDIT =====")
    print(f"Total rows: {report['total']}")
    print("Datasets:")
    for path, count in report["datasets"].items():
        print(f"  {path}: {count}")
    print("Topics:")
    for key, value in sorted(report["topics"].items()):
        print(f"  {key}: {value}")
    print("Ambiguity:")
    for key, value in sorted(report["ambiguity"].items()):
        print(f"  {key}: {value}")
    print("Answer shapes:")
    for key, value in sorted(report["answer_shapes"].items()):
        print(f"  {key}: {value}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
