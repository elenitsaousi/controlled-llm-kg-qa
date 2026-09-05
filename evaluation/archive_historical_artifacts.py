#!/usr/bin/env python3
"""Archive historical experiment artifacts without touching canonical files.

Default mode is a dry run. Use --apply to move files to archive_local while
preserving their relative paths. This is safer than deleting old experiments:
the active branch becomes easier to inspect, but local copies remain available.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Set


PROTECTED_PATHS = {
    ".gitignore",
    "README.md",
    "app.py",
    "data/infineon/graph.ttl",
    "data/infineon/ontology.ttl",
    "data/infineon/schema.json",
    "data/infineon/true_demand_ontology_extracted.owl",
    "data/infineon/true_demand_ontology_extracted.ttl",
    "data/infineon/true_demand_webvowl.json",
    "evaluation/question_sets/true_demand_efficiency_500.json",
    "ranking/final1000_wf_train_ranker_data.json",
    "ranking/models/final1000_wf_ranker_scope_origin.json",
    "ranking/models/final1000_wf_ranker_shape_features.json",
    "ranking/models/final1000_wf_ranker_shortage_grouped.json",
    "results/final1000_wf_test_eval_schema_no_ml.json",
    "results/final1000_wf_test_scope_origin_m010.json",
    "results/final1000_wf_test_entropy_regime_schema_vs_ml.json",
    "results/final1000_wf_test_entropy_regime_schema_vs_ml.md",
    "results/final1000_wf_test_entropy_regime_diagnostics.json",
    "results/final1000_wf_test_entropy_regime_diagnostics.md",
    "results/final1000_wf_test_scope_origin_confidence_routing_sorted_v2.json",
    "results/final1000_wf_test_scope_origin_confidence_routing_sorted_v2.md",
    "results/kgqa_system_accuracy_audit_500.csv",
    "results/kgqa_system_accuracy_audit_500_v2_labeled.csv",
    "results/kgqa_system_accuracy_audit_500_v2_labeled.json",
    "results/kgqa_system_accuracy_audit_500_v2_labeled.md",
    "results/kgqa_system_accuracy_audit_500_v2_review_needed.csv",
    "results/kgqa_efficiency_500_after_direct_report.json",
    "results/kgqa_efficiency_500_after_direct_report.md",
}


ARCHIVE_GLOBS = {
    "legacy_ranker_training_data": [
        "ranking/final360*.json",
        "ranking/infineon_training_data*.json",
    ],
    "legacy_models": [
        "ranking/models/final360*",
        "ranking/models/final1000_wf_ranker.json",
        "ranking/models/final1000_wf_ranker_catalog_status.json",
        "ranking/models/final1000_wf_ranker_output_vars.json",
        "ranking/models/final1000_wf_ranker_projection_grouping.json",
        "ranking/models/final1000_wf_xgb*.pkl",
        "ranking/models/infineon_np_tfidf_ranker_split.json",
        "ranking/models/infineon_query_plan_predictor.json",
        "ranking/models/infineon_ranker.joblib",
        "ranking/models/logistic_ranker.joblib",
    ],
    "legacy_results": [
        "results/infineon_eval_unseen_50_results.json",
        "results/infineon_holdout_eval_50.json",
        "results/infineon_query_plan_predictor_eval.json",
        "results/infineon_test_final*",
        "results/current_kgqa_coverage_audit.json",
        "results/current_kgqa_seed_bank.json",
        "results/final_kgqa_benchmark_*_plan.json",
    ],
    "scratch_webvowl_exports": [
        "true_demand_webvowl.json",
        "true_demand_webvowl_clean.json",
    ],
}


@dataclass(frozen=True)
class Candidate:
    path: Path
    reason: str


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _collect_candidates(root: Path, *, categories: Iterable[str]) -> List[Candidate]:
    protected = set(PROTECTED_PATHS)
    out: Dict[str, Candidate] = {}
    for category in categories:
        for pattern in ARCHIVE_GLOBS.get(category, []):
            for path in root.glob(pattern):
                if not path.is_file():
                    continue
                rel = _rel(path, root)
                if rel in protected:
                    continue
                if rel.startswith("archive_local/"):
                    continue
                out.setdefault(rel, Candidate(path=path, reason=category))
    return [out[key] for key in sorted(out)]


def _manifest(candidates: List[Candidate], root: Path, archive_root: Path, *, applied: bool) -> Dict[str, object]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "archive_root": str(archive_root),
        "applied": applied,
        "count": len(candidates),
        "files": [
            {
                "path": _rel(candidate.path, root),
                "reason": candidate.reason,
                "size_bytes": candidate.path.stat().st_size if candidate.path.exists() else None,
            }
            for candidate in candidates
        ],
    }


def archive_candidates(candidates: List[Candidate], root: Path, archive_root: Path) -> None:
    for candidate in candidates:
        rel = Path(_rel(candidate.path, root))
        target = archive_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Archive target already exists: {target}")
        shutil.move(str(candidate.path), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive historical KGQA experiment artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--archive-root", default="")
    parser.add_argument("--apply", action="store_true", help="Actually move files. Omit for dry-run.")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=sorted(ARCHIVE_GLOBS),
        choices=sorted(ARCHIVE_GLOBS),
        help="Artifact categories to archive.",
    )
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    archive_root = (
        Path(args.archive_root).resolve()
        if args.archive_root
        else root / "archive_local" / f"historical_artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    candidates = _collect_candidates(root, categories=args.categories)
    manifest = _manifest(candidates, root, archive_root, applied=args.apply)

    manifest_path = Path(args.manifest) if args.manifest else root / "results" / "historical_artifact_archive_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.apply:
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_candidates(candidates, root, archive_root)

    print("===== HISTORICAL ARTIFACT ARCHIVE =====")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Candidates: {len(candidates)}")
    print(f"Archive root: {archive_root}")
    print(f"Manifest: {manifest_path}")
    by_reason: Dict[str, int] = {}
    for candidate in candidates:
        by_reason[candidate.reason] = by_reason.get(candidate.reason, 0) + 1
    for reason, count in sorted(by_reason.items()):
        print(f"  {reason}: {count}")
    for candidate in candidates[:80]:
        print(f"  {candidate.reason}: {_rel(candidate.path, root)}")
    if len(candidates) > 80:
        print(f"  ... {len(candidates) - 80} more")


if __name__ == "__main__":
    main()
