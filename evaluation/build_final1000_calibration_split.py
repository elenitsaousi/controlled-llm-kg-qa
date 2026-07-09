#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
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


def _candidate_label(candidate: Dict[str, object]) -> str:
    return str(candidate.get("label", "")).strip().lower()


def _detail_status(detail: Dict[str, object]) -> str:
    candidates = list(detail.get("candidates") or [])
    top1 = bool(detail.get("top1_correct"))
    any_correct = bool(detail.get("any_correct"))
    if top1:
        return "top1_correct"
    if any_correct:
        return "ranking_failure"
    if candidates:
        return "generation_failure"
    return "no_candidates"


def _load_detail_meta(results_path: Optional[str]) -> Dict[str, Dict[str, object]]:
    if not results_path:
        return {}
    payload = _load_json(results_path)
    out: Dict[str, Dict[str, object]] = {}
    for detail in list(payload.get("details") or []):
        qid = str(detail.get("id", "")).strip()
        if not qid:
            continue
        candidates = list(detail.get("candidates") or [])
        out[qid] = {
            "status": _detail_status(detail),
            "candidate_count": len(candidates),
            "correct_rank": next(
                (
                    idx + 1
                    for idx, cand in enumerate(candidates)
                    if _candidate_label(cand) == "correct"
                ),
                None,
            ),
        }
    return out


def _row_family(row: Dict[str, object]) -> str:
    return str(row.get("family") or row.get("topic") or "unknown").strip() or "unknown"


def _strat_key(row: Dict[str, object], detail_meta: Dict[str, Dict[str, object]]) -> Tuple[str, str]:
    qid = str(row.get("id", "")).strip()
    family = _row_family(row)
    status = str(detail_meta.get(qid, {}).get("status") or "unknown")
    return family, status


def _sample_grouped(
    rows: List[Dict[str, object]],
    detail_meta: Dict[str, Dict[str, object]],
    size: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_strat_key(row, detail_meta)].append(row)

    for group_rows in groups.values():
        rng.shuffle(group_rows)

    # Reserve at least one row for every family when possible.
    selected_ids = set()
    selected: List[Dict[str, object]] = []
    by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_family[_row_family(row)].append(row)
    for family in sorted(by_family):
        candidates = list(by_family[family])
        rng.shuffle(candidates)
        for row in candidates:
            qid = str(row.get("id", "")).strip()
            if qid and qid not in selected_ids:
                selected.append(row)
                selected_ids.add(qid)
                break
        if len(selected) >= size:
            return selected[:size]

    total = len(rows)
    quotas: Dict[Tuple[str, str], int] = {}
    remainders: List[Tuple[float, Tuple[str, str]]] = []
    remaining_slots = max(0, size - len(selected))
    for key, group_rows in groups.items():
        exact = remaining_slots * (len(group_rows) / total) if total else 0
        quota = int(exact)
        quotas[key] = min(quota, len(group_rows))
        remainders.append((exact - quota, key))

    for key, quota in quotas.items():
        for row in groups[key]:
            if quota <= 0:
                break
            qid = str(row.get("id", "")).strip()
            if qid and qid not in selected_ids:
                selected.append(row)
                selected_ids.add(qid)
                quota -= 1

    remainders.sort(reverse=True)
    while len(selected) < size:
        added = False
        for _, key in remainders:
            for row in groups[key]:
                qid = str(row.get("id", "")).strip()
                if qid and qid not in selected_ids:
                    selected.append(row)
                    selected_ids.add(qid)
                    added = True
                    break
            if len(selected) >= size:
                break
        if not added:
            break

    rng.shuffle(selected)
    return selected[:size]


def _distribution(rows: List[Dict[str, object]], detail_meta: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    family = Counter(_row_family(row) for row in rows)
    status = Counter(
        str(detail_meta.get(str(row.get("id", "")).strip(), {}).get("status") or "unknown")
        for row in rows
    )
    family_status = Counter(
        "|".join(_strat_key(row, detail_meta))
        for row in rows
    )
    return {
        "family": dict(sorted(family.items())),
        "status": dict(sorted(status.items())),
        "family_status": dict(sorted(family_status.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stratified 200-question calibration split and an untouched "
            "holdout split from the final 1000 KG benchmark."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", default="")
    parser.add_argument("--calibration-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--out-calibration", required=True)
    parser.add_argument("--out-holdout", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()

    rows = _load_json(args.dataset)
    if not isinstance(rows, list):
        raise RuntimeError(f"Dataset must be a JSON list: {args.dataset}")
    if args.calibration_size <= 0 or args.calibration_size >= len(rows):
        raise RuntimeError("--calibration-size must be between 1 and dataset size - 1")

    detail_meta = _load_detail_meta(args.results or None)
    calibration = _sample_grouped(rows, detail_meta, args.calibration_size, args.seed)
    cal_ids = {str(row.get("id", "")).strip() for row in calibration}
    holdout = [row for row in rows if str(row.get("id", "")).strip() not in cal_ids]

    manifest = {
        "dataset": args.dataset,
        "results": args.results or None,
        "seed": args.seed,
        "calibration_size": len(calibration),
        "holdout_size": len(holdout),
        "calibration": _distribution(calibration, detail_meta),
        "holdout": _distribution(holdout, detail_meta),
    }

    _write_json(args.out_calibration, calibration)
    _write_json(args.out_holdout, holdout)
    _write_json(args.out_manifest, manifest)

    print("===== FINAL1000 CALIBRATION SPLIT =====")
    print(f"Dataset: {args.dataset}")
    print(f"Calibration rows: {len(calibration)}")
    print(f"Holdout rows: {len(holdout)}")
    print(f"Seed: {args.seed}")
    print(f"Calibration: {args.out_calibration}")
    print(f"Holdout: {args.out_holdout}")
    print(f"Manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
