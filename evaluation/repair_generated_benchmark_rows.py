#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from evaluation.generate_final_kgqa_benchmark_llm import generate_rows


def _row_index(row_id: str) -> int:
    return int(str(row_id).removeprefix("FINALKGQA")) - 1


def repair_rows(
    plan: Dict[str, object],
    rows: List[Dict[str, object]],
    row_ids: List[str],
    *,
    client,
    request_pause_sec: float = 0.0,
) -> List[Dict[str, object]]:
    repaired = list(rows)
    for row_id in row_ids:
        idx = _row_index(row_id)
        plan_row = dict(plan["rows"][idx])
        generated = generate_rows(
            {"rows": [plan_row]},
            client=client,
            limit=1,
            request_pause_sec=request_pause_sec,
        )
        replacement = dict(generated[-1])
        replacement["id"] = row_id
        repaired[idx] = replacement
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate selected rows in a generated KGQA benchmark.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--request-pause-sec", type=float, default=2.1)
    args = parser.parse_args()

    from llm.client import InfineonGPTClient

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    repaired = repair_rows(
        plan,
        rows,
        args.ids,
        client=InfineonGPTClient(temperature=0.2, max_tokens=300),
        request_pause_sec=args.request_pause_sec,
    )
    Path(args.out).write_text(json.dumps(repaired, indent=2) + "\n", encoding="utf-8")
    print("===== GENERATED BENCHMARK ROW REPAIR =====")
    print(f"Input rows: {len(rows)}")
    print(f"Repaired rows: {len(args.ids)}")
    print(f"IDs: {', '.join(args.ids)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
