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
    if bool(detail.get("top1_correct")):
        return "top1_correct"
    if bool(detail.get("any_correct")):
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


def _distribute_counts(n: int, train_ratio: float, tune_ratio: float) -> Tuple[int, int, int]:
    train = int(round(n * train_ratio))
    tune = int(round(n * tune_ratio))
    train = min(max(train, 0), n)
    tune = min(max(tune, 0), n - train)
    holdout = n - train - tune
    return train, tune, holdout


def _trim_or_fill(
    target_name: str,
    splits: Dict[str, List[Dict[str, object]]],
    target_size: int,
    rng: random.Random,
) -> None:
    while len(splits[target_name]) > target_size:
        donor = splits[target_name].pop()
        receiver = min(
            (name for name in splits if name != target_name),
            key=lambda name: len(splits[name]),
        )
        splits[receiver].append(donor)

    while len(splits[target_name]) < target_size:
        donors = [
            name
            for name in splits
            if name != target_name and len(splits[name]) > 1
        ]
        if not donors:
            break
        donor_name = max(donors, key=lambda name: len(splits[name]))
        idx = rng.randrange(len(splits[donor_name]))
        splits[target_name].append(splits[donor_name].pop(idx))


def _rebalance_exact(
    splits: Dict[str, List[Dict[str, object]]],
    targets: Dict[str, int],
    rng: random.Random,
) -> None:
    while True:
        overfull = [name for name, rows in splits.items() if len(rows) > targets[name]]
        underfull = [name for name, rows in splits.items() if len(rows) < targets[name]]
        if not overfull and not underfull:
            return
        if not overfull or not underfull:
            raise RuntimeError("Could not rebalance split sizes exactly")
        donor_name = max(overfull, key=lambda name: len(splits[name]) - targets[name])
        receiver_name = max(underfull, key=lambda name: targets[name] - len(splits[name]))
        idx = rng.randrange(len(splits[donor_name]))
        splits[receiver_name].append(splits[donor_name].pop(idx))


def split_rows(
    rows: List[Dict[str, object]],
    detail_meta: Dict[str, Dict[str, object]],
    train_size: int,
    tune_size: int,
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    total = len(rows)
    if train_size <= 0 or tune_size <= 0 or train_size + tune_size >= total:
        raise RuntimeError("--train-size and --tune-size must leave a non-empty holdout")

    holdout_size = total - train_size - tune_size
    train_ratio = train_size / total
    tune_ratio = tune_size / total
    rng = random.Random(seed)

    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_strat_key(row, detail_meta)].append(row)

    splits: Dict[str, List[Dict[str, object]]] = {
        "train": [],
        "tune": [],
        "holdout": [],
    }
    for key in sorted(groups):
        group_rows = list(groups[key])
        rng.shuffle(group_rows)
        train_n, tune_n, _ = _distribute_counts(len(group_rows), train_ratio, tune_ratio)
        splits["train"].extend(group_rows[:train_n])
        splits["tune"].extend(group_rows[train_n : train_n + tune_n])
        splits["holdout"].extend(group_rows[train_n + tune_n :])

    for name, target in [
        ("train", train_size),
        ("tune", tune_size),
        ("holdout", holdout_size),
    ]:
        _trim_or_fill(name, splits, target, rng)

    _rebalance_exact(
        splits,
        {
            "train": train_size,
            "tune": tune_size,
            "holdout": holdout_size,
        },
        rng,
    )

    for split_rows_ in splits.values():
        rng.shuffle(split_rows_)

    seen = set()
    for name, split_rows_ in splits.items():
        for row in split_rows_:
            qid = str(row.get("id", "")).strip()
            if not qid:
                raise RuntimeError(f"Row without id in {name}")
            if qid in seen:
                raise RuntimeError(f"Question id appears in more than one split: {qid}")
            seen.add(qid)
    if len(seen) != total:
        raise RuntimeError("Split did not preserve all dataset rows")
    return splits


def _distribution(rows: List[Dict[str, object]], detail_meta: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    family = Counter(_row_family(row) for row in rows)
    status = Counter(
        str(detail_meta.get(str(row.get("id", "")).strip(), {}).get("status") or "unknown")
        for row in rows
    )
    family_status = Counter("|".join(_strat_key(row, detail_meta)) for row in rows)
    return {
        "size": len(rows),
        "family": dict(sorted(family.items())),
        "status": dict(sorted(status.items())),
        "family_status": dict(sorted(family_status.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create non-overlapping train/tune/holdout splits for the repaired "
            "final-1000 KG benchmark. The tune split is for threshold selection; "
            "the holdout split is the only clean model-selection evidence."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", default="")
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--tune-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-tune", required=True)
    parser.add_argument("--out-holdout", required=True)
    parser.add_argument("--out-manifest", required=True)
    args = parser.parse_args()

    rows = _load_json(args.dataset)
    if not isinstance(rows, list):
        raise RuntimeError(f"Dataset must be a JSON list: {args.dataset}")

    detail_meta = _load_detail_meta(args.results or None)
    splits = split_rows(
        rows=rows,
        detail_meta=detail_meta,
        train_size=args.train_size,
        tune_size=args.tune_size,
        seed=args.seed,
    )

    manifest = {
        "dataset": args.dataset,
        "results": args.results or None,
        "seed": args.seed,
        "train_size": len(splits["train"]),
        "tune_size": len(splits["tune"]),
        "holdout_size": len(splits["holdout"]),
        "notes": [
            "Train is used to fit the reranker.",
            "Tune is used only to select guarded reranking thresholds.",
            "Holdout is not used for training or threshold selection.",
        ],
        "splits": {
            name: _distribution(split_rows_, detail_meta)
            for name, split_rows_ in splits.items()
        },
        "ids": {
            name: [str(row.get("id", "")).strip() for row in split_rows_]
            for name, split_rows_ in splits.items()
        },
    }

    _write_json(args.out_train, splits["train"])
    _write_json(args.out_tune, splits["tune"])
    _write_json(args.out_holdout, splits["holdout"])
    _write_json(args.out_manifest, manifest)

    print("===== FINAL1000 REPAIRED TRAIN/TUNE/HOLDOUT SPLIT =====")
    print(f"Dataset: {args.dataset}")
    print(f"Results for stratification: {args.results or 'none'}")
    print(f"Seed: {args.seed}")
    print(f"Train rows: {len(splits['train'])}")
    print(f"Tune rows: {len(splits['tune'])}")
    print(f"Holdout rows: {len(splits['holdout'])}")
    print(f"Train: {args.out_train}")
    print(f"Tune: {args.out_tune}")
    print(f"Holdout: {args.out_holdout}")
    print(f"Manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
