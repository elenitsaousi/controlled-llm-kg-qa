#!/usr/bin/env python3
"""
Create leakage-safe train/dev/test splits for Infineon benchmark questions.

Rules:
- grouping by query family (no family appears in more than one split)
- approximate stratification by ambiguity label (low/mid/high)
- configurable split ratios
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


VAR_RE = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
SINGLE_QUOTE_STR_RE = re.compile(r"'[^']*'")
DOUBLE_QUOTE_STR_RE = re.compile(r'"[^"]*"')
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _query_family_signature(query: str) -> str:
    q = " ".join(query.strip().split())
    q = SINGLE_QUOTE_STR_RE.sub("'STR'", q)
    q = DOUBLE_QUOTE_STR_RE.sub('"STR"', q)
    q = NUMBER_RE.sub("NUM", q)
    q = VAR_RE.sub("?VAR", q)
    return "fam_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:16]


def _normalize_label(label: str) -> str:
    x = (label or "").strip().lower()
    if x == "medium":
        return "mid"
    return x or "unknown"


def _parse_ratios(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--ratios must have exactly 3 comma-separated values: train,dev,test")
    vals = tuple(float(x) for x in parts)
    if any(v <= 0 for v in vals):
        raise ValueError("All split ratios must be > 0")
    s = sum(vals)
    return vals[0] / s, vals[1] / s, vals[2] / s


def _group_items(items: Sequence[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for idx, item in enumerate(items, start=1):
        query = str(item.get("query", "")).strip()
        family = str(item.get("family", "")).strip() or _query_family_signature(query)
        normalized = dict(item)
        normalized["family"] = family
        normalized["ambiguity_label"] = _normalize_label(str(item.get("ambiguity_label", "")))
        if not normalized.get("id"):
            normalized["id"] = f"Q{idx}"
        groups[family].append(normalized)
    return groups


def _label_counts(rows: Sequence[Dict]) -> Counter:
    c = Counter()
    for r in rows:
        c[_normalize_label(str(r.get("ambiguity_label", "")))] += 1
    return c


def _build_splits(
    groups: Dict[str, List[Dict]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, List[Dict]]:
    split_names = ["train", "dev", "test"]
    total_n = sum(len(v) for v in groups.values())
    total_labels = Counter()
    for rows in groups.values():
        total_labels.update(_label_counts(rows))

    target_total = {
        split_names[i]: ratios[i] * total_n for i in range(3)
    }
    target_families = {
        split_names[i]: ratios[i] * len(groups) for i in range(3)
    }
    target_labels = {
        split: {lab: ratios[idx] * total_labels[lab] for lab in total_labels}
        for idx, split in enumerate(split_names)
    }

    fam_records = []
    for fam, rows in groups.items():
        fam_records.append((fam, rows, _label_counts(rows), len(rows)))

    rnd = random.Random(seed)
    rnd.shuffle(fam_records)
    fam_records.sort(key=lambda x: x[3], reverse=True)

    out = {name: [] for name in split_names}
    counts_total = {name: 0 for name in split_names}
    counts_label = {name: Counter() for name in split_names}
    counts_families = {name: 0 for name in split_names}

    for fam, rows, fam_lab_counts, fam_n in fam_records:
        best_split = None
        best_score = None
        for split in split_names:
            need_total = max(0.0, target_total[split] - counts_total[split])
            contrib_total = min(float(fam_n), need_total)
            overshoot_total = max(
                0.0, float(counts_total[split] + fam_n) - target_total[split]
            )
            need_family = max(0.0, target_families[split] - counts_families[split])
            contrib_family = min(1.0, need_family)
            overshoot_family = max(
                0.0, float(counts_families[split] + 1) - target_families[split]
            )

            contrib_label = 0.0
            overshoot_label = 0.0
            for lab in total_labels:
                need_lab = max(0.0, target_labels[split][lab] - counts_label[split][lab])
                add_lab = float(fam_lab_counts.get(lab, 0))
                contrib_label += min(add_lab, need_lab)
                overshoot_label += max(
                    0.0,
                    float(counts_label[split][lab] + fam_lab_counts.get(lab, 0))
                    - target_labels[split][lab],
                )

            # Higher score is better: reward filling deficits, penalize overshoot.
            score = (
                3.0 * contrib_total
                + 1.5 * contrib_label
                + 1.0 * contrib_family
                - 0.8 * overshoot_total
                - 0.4 * overshoot_label
                - 0.3 * overshoot_family
            )

            # Tie-breaker: prefer split with larger remaining capacity.
            score += 1e-6 * need_total

            if best_score is None or score > best_score:
                best_score = score
                best_split = split

        assert best_split is not None
        out[best_split].extend(rows)
        counts_total[best_split] += fam_n
        counts_families[best_split] += 1
        for lab, c in fam_lab_counts.items():
            counts_label[best_split][lab] += c

    # Stable ordering within each split for deterministic diffs.
    for split in split_names:
        out[split].sort(key=lambda x: str(x.get("id", "")))
    return out


def _validate_no_family_leakage(splits: Dict[str, List[Dict]]) -> None:
    fam_sets = {}
    for split, rows in splits.items():
        fam_sets[split] = {str(r.get("family", "")) for r in rows}
    names = list(fam_sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            overlap = fam_sets[a] & fam_sets[b]
            if overlap:
                sample = sorted(list(overlap))[:5]
                raise RuntimeError(
                    f"Family leakage detected between {a} and {b}. Sample: {sample}"
                )


def _split_stats(rows: Sequence[Dict]) -> Dict[str, object]:
    labels = Counter(_normalize_label(str(r.get("ambiguity_label", ""))) for r in rows)
    families = {str(r.get("family", "")) for r in rows}
    return {
        "questions": len(rows),
        "families": len(families),
        "labels": dict(labels),
    }


def run(
    dataset_path: Path,
    out_dir: Path,
    ratios: Tuple[float, float, float],
    seed: int,
) -> None:
    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Invalid or empty dataset file: {dataset_path}")

    groups = _group_items(items)
    splits = _build_splits(groups=groups, ratios=ratios, seed=seed)
    _validate_no_family_leakage(splits)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_files = {
        "train": out_dir / "train.json",
        "dev": out_dir / "dev.json",
        "test": out_dir / "test.json",
    }
    for split, path in out_files.items():
        with open(path, "w", encoding="utf-8") as f:
            json.dump(splits[split], f, indent=2, ensure_ascii=False)
            f.write("\n")

    manifest = {
        "dataset": str(dataset_path),
        "ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
        "seed": seed,
        "total_questions": len(items),
        "total_families": len(groups),
        "splits": {
            split: _split_stats(rows) for split, rows in splits.items()
        },
        "files": {k: str(v) for k, v in out_files.items()},
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("Saved splits:")
    for split in ("train", "dev", "test"):
        stats = manifest["splits"][split]
        print(
            f"  {split}: questions={stats['questions']} families={stats['families']} "
            f"labels={stats['labels']}"
        )
    print(f"Manifest: {out_dir / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create grouped stratified train/dev/test splits for Infineon dataset."
    )
    parser.add_argument("--dataset", default="data/infineon/infineon_dataset_500.json")
    parser.add_argument("--out-dir", default="data/infineon/splits/infineon_500")
    parser.add_argument(
        "--ratios",
        default="0.8,0.1,0.1",
        help="train,dev,test ratios (comma-separated).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        dataset_path=Path(args.dataset),
        out_dir=Path(args.out_dir),
        ratios=_parse_ratios(args.ratios),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
