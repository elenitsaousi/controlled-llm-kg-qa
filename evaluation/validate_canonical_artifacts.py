#!/usr/bin/env python3
"""Validate that the current thesis artifacts are present and internally sane.

This script does not re-run expensive LLM evaluations. It checks that the files
used in the final narrative exist, can be parsed, and contain metrics consistent
with the expected selection/system-evaluation framing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_ARTIFACTS = {
    "schema": "data/infineon/schema.json",
    "graph": "data/infineon/graph.ttl",
    "webvowl": "data/infineon/true_demand_webvowl.json",
    "efficiency_questions": "evaluation/question_sets/true_demand_efficiency_500.json",
    "final_model": "ranking/models/final1000_wf_ranker_scope_origin.json",
    "training_data": "ranking/final1000_wf_train_ranker_data.json",
    "selection_baseline": "results/final1000_wf_test_eval_schema_no_ml.json",
    "selection_ml": "results/final1000_wf_test_scope_origin_m010.json",
    "entropy_comparison": "results/final1000_wf_test_entropy_regime_schema_vs_ml.json",
    "entropy_diagnostics": "results/final1000_wf_test_entropy_regime_diagnostics.json",
    "confidence_routing": "results/final1000_wf_test_scope_origin_confidence_routing_sorted_v2.json",
    "system_audit": "results/kgqa_system_accuracy_audit_500_v2_labeled.csv",
    "system_efficiency": "results/kgqa_efficiency_500_after_direct_report.json",
}


def _load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _label(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("label", "") or "").strip().lower()


def _details(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    return [row for row in list(payload.get("details") or []) if isinstance(row, dict)]


def _candidates(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in list(detail.get("candidates") or []) if isinstance(c, dict)]


def _top1_correct(detail: Dict[str, Any]) -> bool:
    if isinstance(detail.get("top1_correct"), bool):
        return bool(detail["top1_correct"])
    candidates = _candidates(detail)
    return bool(candidates and _label(candidates[0]) == "correct")


def _any_correct(detail: Dict[str, Any]) -> bool:
    if isinstance(detail.get("any_correct"), bool):
        return bool(detail["any_correct"])
    return any(_label(candidate) == "correct" for candidate in _candidates(detail))


def _result_summary(path: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json(path)
    rows = _details(payload)
    if not rows:
        return None
    top1 = sum(1 for row in rows if _top1_correct(row))
    any_correct = sum(1 for row in rows if _any_correct(row))
    return {
        "path": str(path),
        "questions": len(rows),
        "top1_correct": top1,
        "top1_accuracy": top1 / len(rows),
        "any_correct": any_correct,
        "any_correct_rate": any_correct / len(rows),
    }


def _audit_label(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"correct", "yes", "y", "1", "true", "ok"}:
        return "correct"
    if text in {"incorrect", "wrong", "no", "n", "0", "false"}:
        return "incorrect"
    if text in {"unclear", "ambiguous", "partial", "unknown", "?"}:
        return "unclear"
    return "unlabeled"


def _system_audit_summary(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if not rows:
        return None

    overall: Counter = Counter()
    by_mode: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        label = _audit_label(row.get("correctness"))
        mode = str(row.get("system_mode") or "unknown")
        overall[label] += 1
        by_mode[mode][label] += 1

    def metrics(counter: Counter) -> Dict[str, Any]:
        labeled = counter["correct"] + counter["incorrect"] + counter["unclear"]
        denom = labeled
        return {
            "labeled": labeled,
            "correct": counter["correct"],
            "incorrect": counter["incorrect"],
            "unclear": counter["unclear"],
            "unlabeled": counter["unlabeled"],
            "accuracy_unclear_as_incorrect": counter["correct"] / denom if denom else 0.0,
        }

    return {
        "path": str(path),
        "rows": len(rows),
        "overall": metrics(overall),
        "by_mode": {mode: metrics(counter) for mode, counter in sorted(by_mode.items())},
    }


def _check_model(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return {"path": str(path), "ok": False, "reason": "cannot_parse_json"}
    feature_names = list(payload.get("feature_names") or [])
    weights = list(payload.get("weights") or [])
    mean = list(payload.get("scaler_mean") or [])
    std = list(payload.get("scaler_std") or [])
    expected = len(feature_names)
    ok = bool(expected and len(weights) == expected and len(mean) == expected and len(std) == expected)
    return {
        "path": str(path),
        "ok": ok,
        "feature_count": expected,
        "weights": len(weights),
        "scaler_mean": len(mean),
        "scaler_std": len(std),
        "model_type": payload.get("model_type"),
    }


def _check_entropy_diagnostics(path: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    subsets = payload.get("subsets") if isinstance(payload.get("subsets"), dict) else {}
    out: Dict[str, Any] = {"path": str(path), "subsets": {}}
    for name, subset in subsets.items():
        if not isinstance(subset, dict):
            continue
        summary = subset.get("summary") if isinstance(subset.get("summary"), dict) else {}
        category_counts = subset.get("category_counts") or []
        out["subsets"][name] = {
            "count": summary.get("count"),
            "avg_entropy": summary.get("avg_entropy"),
            "avg_margin": summary.get("avg_margin"),
            "top_categories": category_counts[:8],
        }
    interpretation = payload.get("diagnostic_interpretation")
    if isinstance(interpretation, dict):
        out["supported_claims"] = interpretation.get("supported_claims") or []
        out["caveats"] = interpretation.get("caveats") or []
    return out


def _check_entropy_comparison(path: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    regimes = payload.get("by_entropy_regime") if isinstance(payload.get("by_entropy_regime"), list) else []
    return {
        "path": str(path),
        "total_with_scores": summary.get("total_with_scores"),
        "baseline_accuracy": summary.get("baseline_accuracy"),
        "ml_accuracy": summary.get("ml_accuracy"),
        "delta_accuracy": summary.get("delta_accuracy"),
        "by_entropy_regime": [
            {
                "regime": row.get("regime"),
                "count": row.get("count"),
                "baseline_accuracy": row.get("baseline_accuracy"),
                "ml_accuracy": row.get("ml_accuracy"),
                "delta_accuracy": row.get("delta_accuracy"),
            }
            for row in regimes
            if isinstance(row, dict)
        ],
    }


def _check_efficiency(path: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "path": str(path),
        "total_queries": summary.get("total_queries") or summary.get("queries"),
        "llm_calls": summary.get("llm_calls"),
        "estimated_cost": summary.get("estimated_cost"),
        "baseline_cost": summary.get("baseline_cost"),
        "estimated_savings": summary.get("estimated_savings"),
        "cost_reduction_pct": summary.get("cost_reduction_pct"),
    }


def _check_question_set(path: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        return None
    topics = Counter(str(row.get("topic") or "unknown") for row in payload if isinstance(row, dict))
    return {"path": str(path), "questions": len(payload), "topics": topics.most_common()}


def _exists_check(paths: Dict[str, str], root: Path) -> List[Dict[str, Any]]:
    rows = []
    for name, rel in paths.items():
        path = root / rel
        rows.append({"name": name, "path": rel, "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0})
    return rows


def _status(ok: bool, message: str, severity: str = "error") -> Dict[str, str]:
    return {"status": "ok" if ok else severity, "message": message}


def _sanity_checks(report: Dict[str, Any]) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []
    exists = {row["name"]: bool(row["exists"]) for row in report.get("presence", [])}
    for required in ("schema", "graph", "webvowl", "efficiency_questions", "final_model", "training_data"):
        checks.append(_status(exists.get(required, False), f"Required artifact exists: {required}"))

    model = report.get("model") or {}
    checks.append(_status(bool(model.get("ok")), "Final ranker model has matching feature/weight/scaler dimensions."))

    baseline = report.get("selection_baseline")
    ml = report.get("selection_ml")
    if baseline and ml:
        checks.append(_status(baseline.get("questions") == ml.get("questions"), "Baseline and ML selection files have the same question count."))
        checks.append(_status((ml.get("top1_accuracy") or 0.0) > (baseline.get("top1_accuracy") or 0.0), "ML selection accuracy is higher than baseline selection accuracy."))
        checks.append(_status((ml.get("any_correct_rate") or 0.0) >= 0.85, "ML result keeps high Any-Correct candidate recall.", "warning"))
    else:
        checks.append(_status(False, "Selection baseline or ML result is missing; selection claims cannot be validated here.", "warning"))

    entropy = report.get("entropy_comparison")
    if entropy:
        checks.append(_status((entropy.get("ml_accuracy") or 0.0) > (entropy.get("baseline_accuracy") or 0.0), "Entropy comparison shows ML improves selection over baseline."))
    else:
        checks.append(_status(False, "Entropy comparison file missing or unreadable.", "warning"))

    diagnostics = report.get("entropy_diagnostics")
    if diagnostics:
        high = dict((diagnostics.get("subsets") or {}).get("high_entropy_correct") or {})
        high_count = high.get("count") or 0
        high_categories = dict(high.get("top_categories") or [])
        harmless = high_categories.get("harmless_high_entropy_near_duplicate", 0)
        checks.append(_status(not high_count or harmless / high_count >= 0.5, "High-entropy correct cases are mostly near-duplicate/harmless uncertainty.", "warning"))
    else:
        checks.append(_status(False, "Entropy diagnostics file missing or unreadable.", "warning"))

    audit = report.get("system_audit")
    if audit:
        overall = audit.get("overall") or {}
        checks.append(_status(audit.get("rows") == 500, "System audit contains 500 rows."))
        checks.append(_status(overall.get("unlabeled") == 0, "System audit has no unlabeled rows.", "warning"))
        checks.append(_status((overall.get("accuracy_unclear_as_incorrect") or 0.0) >= 0.75, "System-level accuracy is at least 75%.", "warning"))
    else:
        checks.append(_status(False, "System audit file missing or unreadable.", "warning"))

    efficiency = report.get("system_efficiency")
    if efficiency:
        checks.append(_status((efficiency.get("cost_reduction_pct") or 0.0) >= 50.0, "Efficiency report shows at least 50% cost reduction.", "warning"))
    else:
        checks.append(_status(False, "System efficiency report missing or unreadable.", "warning"))

    return checks


def build_report(root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "root": str(root),
        "presence": _exists_check(DEFAULT_ARTIFACTS, root),
    }
    report["model"] = _check_model(root / DEFAULT_ARTIFACTS["final_model"]) if (root / DEFAULT_ARTIFACTS["final_model"]).exists() else None
    report["efficiency_questions"] = _check_question_set(root / DEFAULT_ARTIFACTS["efficiency_questions"]) if (root / DEFAULT_ARTIFACTS["efficiency_questions"]).exists() else None
    report["selection_baseline"] = _result_summary(root / DEFAULT_ARTIFACTS["selection_baseline"]) if (root / DEFAULT_ARTIFACTS["selection_baseline"]).exists() else None
    report["selection_ml"] = _result_summary(root / DEFAULT_ARTIFACTS["selection_ml"]) if (root / DEFAULT_ARTIFACTS["selection_ml"]).exists() else None
    report["entropy_comparison"] = _check_entropy_comparison(root / DEFAULT_ARTIFACTS["entropy_comparison"]) if (root / DEFAULT_ARTIFACTS["entropy_comparison"]).exists() else None
    report["entropy_diagnostics"] = _check_entropy_diagnostics(root / DEFAULT_ARTIFACTS["entropy_diagnostics"]) if (root / DEFAULT_ARTIFACTS["entropy_diagnostics"]).exists() else None
    report["system_audit"] = _system_audit_summary(root / DEFAULT_ARTIFACTS["system_audit"]) if (root / DEFAULT_ARTIFACTS["system_audit"]).exists() else None
    report["system_efficiency"] = _check_efficiency(root / DEFAULT_ARTIFACTS["system_efficiency"]) if (root / DEFAULT_ARTIFACTS["system_efficiency"]).exists() else None
    report["checks"] = _sanity_checks(report)
    return report


def _fmt_pct(value: object) -> str:
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "-"


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Canonical Artifact Validation",
        "",
        "This report checks artifact presence and internal consistency. It does not replace a manual review of answer correctness.",
        "",
        "## Checks",
        "",
        "| Status | Message |",
        "|---|---|",
    ]
    for row in report.get("checks") or []:
        lines.append(f"| `{row.get('status')}` | {row.get('message')} |")

    lines.extend(["", "## Presence", "", "| Name | Exists | Size | Path |", "|---|---:|---:|---|"])
    for row in report.get("presence") or []:
        lines.append(f"| `{row['name']}` | {row['exists']} | {row['size_bytes']} | `{row['path']}` |")

    baseline = report.get("selection_baseline")
    ml = report.get("selection_ml")
    if baseline or ml:
        lines.extend(["", "## Selection Results", "", "| File | Questions | Top-1 | Any Correct |", "|---|---:|---:|---:|"])
        for name, row in (("baseline", baseline), ("ml", ml)):
            if row:
                lines.append(
                    f"| `{name}` | {row['questions']} | {row['top1_correct']} ({_fmt_pct(row['top1_accuracy'])}) | "
                    f"{row['any_correct']} ({_fmt_pct(row['any_correct_rate'])}) |"
                )

    entropy = report.get("entropy_comparison")
    if entropy:
        lines.extend(["", "## Entropy Comparison", "", "| Metric | Value |", "|---|---:|"])
        lines.append(f"| Baseline accuracy | {_fmt_pct(entropy.get('baseline_accuracy'))} |")
        lines.append(f"| ML accuracy | {_fmt_pct(entropy.get('ml_accuracy'))} |")
        lines.append(f"| Delta | {_fmt_pct(entropy.get('delta_accuracy'))} |")
        lines.extend(["", "| Regime | Count | Baseline | ML | Delta |", "|---|---:|---:|---:|---:|"])
        for row in entropy.get("by_entropy_regime") or []:
            lines.append(
                f"| `{row.get('regime')}` | {row.get('count')} | {_fmt_pct(row.get('baseline_accuracy'))} | "
                f"{_fmt_pct(row.get('ml_accuracy'))} | {_fmt_pct(row.get('delta_accuracy'))} |"
            )

    diagnostics = report.get("entropy_diagnostics")
    if diagnostics:
        lines.extend(["", "## Entropy Diagnostics", "", "| Subset | Count | Avg H | Avg Margin | Top Categories |", "|---|---:|---:|---:|---|"])
        for subset, row in (diagnostics.get("subsets") or {}).items():
            cats = ", ".join(f"{name}:{count}" for name, count in row.get("top_categories", [])[:4])
            lines.append(
                f"| `{subset}` | {row.get('count')} | {row.get('avg_entropy', 0):.3f} | "
                f"{row.get('avg_margin', 0):.3f} | {cats} |"
            )

    audit = report.get("system_audit")
    if audit:
        overall = audit.get("overall") or {}
        lines.extend(["", "## System Accuracy Audit", "", "| Rows | Correct | Incorrect | Unclear | Unlabeled | Accuracy |", "|---:|---:|---:|---:|---:|---:|"])
        lines.append(
            f"| {audit.get('rows')} | {overall.get('correct')} | {overall.get('incorrect')} | "
            f"{overall.get('unclear')} | {overall.get('unlabeled')} | "
            f"{_fmt_pct(overall.get('accuracy_unclear_as_incorrect'))} |"
        )
        lines.extend(["", "| Mode | Correct | Incorrect | Unclear | Accuracy |", "|---|---:|---:|---:|---:|"])
        for mode, row in (audit.get("by_mode") or {}).items():
            lines.append(
                f"| `{mode}` | {row.get('correct')} | {row.get('incorrect')} | {row.get('unclear')} | "
                f"{_fmt_pct(row.get('accuracy_unclear_as_incorrect'))} |"
            )

    efficiency = report.get("system_efficiency")
    if efficiency:
        lines.extend(["", "## System Efficiency", "", "| Metric | Value |", "|---|---:|"])
        for key in ("total_queries", "llm_calls", "estimated_cost", "baseline_cost", "estimated_savings", "cost_reduction_pct"):
            lines.append(f"| `{key}` | {efficiency.get(key)} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `ok` means the artifact passed a mechanical consistency check.",
            "- `warning` means the artifact may still be usable, but the related thesis claim should be checked manually.",
            "- This validation cannot prove that every answer is semantically correct; it verifies that the files support the reported evaluation framing.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical KGQA thesis artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-json", default="results/canonical_artifact_validation.json")
    parser.add_argument("--out-md", default="results/canonical_artifact_validation.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    report = build_report(root)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")

    counts = Counter(row["status"] for row in report["checks"])
    print("===== CANONICAL ARTIFACT VALIDATION =====")
    print(f"Root: {root}")
    print(f"Checks: ok={counts['ok']}, warning={counts['warning']}, error={counts['error']}")
    for row in report["checks"]:
        if row["status"] != "ok":
            print(f"{row['status'].upper()}: {row['message']}")
    print(f"JSON: {args.out_json}")
    print(f"Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
