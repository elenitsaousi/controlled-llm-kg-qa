#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def apply_overrides(rows: List[Dict[str, object]], overrides: Dict[str, str]) -> List[Dict[str, object]]:
    out = []
    seen = set()
    for row in rows:
        updated = dict(row)
        row_id = str(updated.get("id") or "")
        if row_id in overrides:
            updated["question"] = str(overrides[row_id]).strip()
            updated["qc_override"] = True
            seen.add(row_id)
        out.append(updated)
    missing = sorted(set(overrides) - seen)
    if missing:
        raise ValueError(f"Override IDs not present in dataset: {', '.join(missing)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply curated question overrides to a generated KGQA benchmark.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
    updated = apply_overrides(rows, overrides)
    Path(args.out).write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print("===== BENCHMARK QUESTION OVERRIDES =====")
    print(f"Input rows: {len(rows)}")
    print(f"Overrides applied: {len(overrides)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
