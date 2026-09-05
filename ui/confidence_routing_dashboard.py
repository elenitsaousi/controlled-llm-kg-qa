#!/usr/bin/env python3
"""Streamlit dashboard for confidence-aware KGQA routing reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency guard
    raise SystemExit(
        "Streamlit is not installed. Install it with:\n\n"
        "  python -m pip install streamlit\n\n"
        "Then run:\n\n"
        "  streamlit run ui/confidence_routing_dashboard.py"
    ) from exc


DEFAULT_REPORT = "results/final1000_wf_test_scope_origin_confidence_routing_safety.json"


def _load_report(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Report must be a JSON object.")
    return payload


def _as_cases(report: Dict[str, object], bucket: str) -> List[Dict[str, object]]:
    if bucket == "low_confidence_examples":
        return [dict(row) for row in report.get("low_confidence_examples") or []]
    if bucket == "high_confidence_wrong_examples":
        return [dict(row) for row in report.get("high_confidence_wrong_examples") or []]
    return []


def _summary_value(report: Dict[str, object], key: str, default: object = 0) -> object:
    return dict(report.get("summary") or {}).get(key, default)


def _policy_bucket(report: Dict[str, object], name: str) -> Dict[str, object]:
    return dict(dict(report.get("policy_buckets") or {}).get(name) or {})


def _fmt_pct(value: object) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _score(row: Dict[str, object], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _all_flags(cases: Iterable[Dict[str, object]]) -> List[str]:
    flags = set()
    for row in cases:
        for flag in row.get("safety_flags") or []:
            flags.add(str(flag))
    return sorted(flags)


def _top3_table(row: Dict[str, object]) -> List[Dict[str, object]]:
    table: List[Dict[str, object]] = []
    for candidate in row.get("top3") or []:
        if not isinstance(candidate, dict):
            continue
        table.append(
            {
                "rank": candidate.get("rank"),
                "score": round(float(candidate.get("score") or 0.0), 4),
                "label": candidate.get("label"),
                "source": candidate.get("source"),
                "query": candidate.get("query"),
            }
        )
    return table


def _distribution_rows(values: object) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in values or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            rows.append({"value": item[0], "count": item[1]})
    return rows


def render_summary(report: Dict[str, object]) -> None:
    summary = dict(report.get("summary") or {})
    inputs = dict(report.get("inputs") or {})
    auto = _policy_bucket(report, "auto_answer")
    clarification = _policy_bucket(report, "clarification")

    st.subheader("System Metrics")
    cols = st.columns(5)
    cols[0].metric(
        "Forced Top1",
        f"{summary.get('forced_top1_correct', 0)}/{summary.get('total', 0)}",
        _fmt_pct(summary.get("forced_top1_accuracy")),
    )
    cols[1].metric(
        "Any Correct",
        f"{summary.get('any_correct', 0)}/{summary.get('total', 0)}",
        _fmt_pct(summary.get("any_correct_rate")),
    )
    cols[2].metric(
        "Auto-answer",
        f"{auto.get('correct', 0)}/{auto.get('count', 0)}",
        _fmt_pct(auto.get("accuracy")),
    )
    cols[3].metric("Auto coverage", str(auto.get("count", 0)), _fmt_pct((auto.get("count", 0) or 0) / max(1, summary.get("total", 1))))
    cols[4].metric(
        "Clarification Any",
        f"{clarification.get('any_correct', 0)}/{clarification.get('count', 0)}",
        _fmt_pct(clarification.get("any_correct_rate")),
    )

    st.caption(
        "Policy: "
        f"score >= {float(inputs.get('policy_min_score') or 0.0):.2f}, "
        f"margin >= {float(inputs.get('policy_min_margin') or 0.0):.2f}, "
        f"safety guard = {inputs.get('enable_safety_guard')}"
    )


def render_distributions(report: Dict[str, object]) -> None:
    st.subheader("Bucket Composition")
    bucket_name = st.segmented_control("Bucket", ["auto_answer", "clarification"], default="auto_answer")
    bucket = _policy_bucket(report, str(bucket_name))
    dimensions = [
        ("families", "Families"),
        ("aggregation", "Aggregation"),
        ("scopes", "Scopes"),
        ("dimensions", "Dimensions"),
        ("answer_shape", "Answer Shape"),
        ("safety_flags", "Safety Flags"),
    ]
    cols = st.columns(2)
    for idx, (key, label) in enumerate(dimensions):
        with cols[idx % 2]:
            st.markdown(f"**{label}**")
            st.dataframe(_distribution_rows(bucket.get(key)), use_container_width=True, hide_index=True)


def render_cases(report: Dict[str, object]) -> None:
    st.subheader("Case Browser")
    source = st.selectbox(
        "Case set",
        ["high_confidence_wrong_examples", "low_confidence_examples"],
        index=0,
    )
    cases = _as_cases(report, source)
    if not cases:
        st.info("No cases available in this report.")
        return

    family_options = sorted({str(row.get("family") or "unknown") for row in cases})
    selected_families = st.multiselect("Families", family_options, default=family_options)
    only_wrong = st.checkbox("Only top1 wrong", value=(source == "high_confidence_wrong_examples"))
    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
    flag_options = _all_flags(cases)
    selected_flags = st.multiselect("Safety flags", flag_options)

    filtered: List[Dict[str, object]] = []
    for row in cases:
        if str(row.get("family") or "unknown") not in selected_families:
            continue
        if only_wrong and row.get("top1_correct"):
            continue
        if _score(row, "score1") < min_score:
            continue
        row_flags = {str(flag) for flag in row.get("safety_flags") or []}
        if selected_flags and not row_flags.intersection(selected_flags):
            continue
        filtered.append(row)

    st.caption(f"Showing {len(filtered)} / {len(cases)} cases")
    for row in filtered:
        title = f"{row.get('id')} | score={_score(row, 'score1'):.3f} | margin={_score(row, 'margin'):.3f}"
        with st.expander(title, expanded=False):
            st.markdown(f"**Question:** {row.get('question')}")
            cols = st.columns(4)
            cols[0].metric("Score1", f"{_score(row, 'score1'):.4f}")
            cols[1].metric("Score2", f"{_score(row, 'score2'):.4f}")
            cols[2].metric("Margin", f"{_score(row, 'margin'):.4f}")
            cols[3].metric("Top1 correct", str(bool(row.get("top1_correct"))))
            st.markdown(f"**Safety flags:** `{row.get('safety_flags') or []}`")
            st.markdown("**Top interpretations**")
            st.dataframe(_top3_table(row), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="KGQA Confidence Routing", layout="wide")
    st.title("KGQA Confidence Routing Dashboard")

    default_path = DEFAULT_REPORT if Path(DEFAULT_REPORT).exists() else ""
    path = st.sidebar.text_input("Routing report JSON", value=default_path)
    if not path:
        st.info("Enter a confidence routing JSON report path.")
        return
    try:
        report = _load_report(path)
    except Exception as exc:
        st.error(f"Could not load report: {exc}")
        return

    render_summary(report)
    st.divider()
    render_distributions(report)
    st.divider()
    render_cases(report)


if __name__ == "__main__":
    main()
