#!/usr/bin/env python3
"""
Generate a 500-question Infineon benchmark from the validated 100-question set.

Design goals:
- keep all queries strictly Infineon/KG grounded
- preserve ambiguity distribution (low/mid/high)
- add NL variability (question paraphrases) while keeping gold queries unchanged
- assign stable query-family ids to prevent train/test leakage in grouped splits
- optionally validate all queries against the Infineon graph before writing output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from rdflib import Graph


VAR_RE = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
SINGLE_QUOTE_STR_RE = re.compile(r"'[^']*'")
DOUBLE_QUOTE_STR_RE = re.compile(r'"[^"]*"')
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

PREFIX = """\
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SEED = BASE_DIR / "data" / "infineon" / "infineon_dataset_100.json"
DEFAULT_GRAPH = BASE_DIR / "data" / "infineon" / "graph.ttl"
DEFAULT_OUT = BASE_DIR / "data" / "infineon" / "infineon_dataset_500.json"


def _ensure_qmark(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return text
    if text.endswith("?"):
        return text
    return text.rstrip(".") + "?"


def _lower_first(text: str) -> str:
    if not text:
        return text
    return text[:1].lower() + text[1:]


def _query_family_signature(query: str) -> str:
    q = " ".join(query.strip().split())
    q = SINGLE_QUOTE_STR_RE.sub("'STR'", q)
    q = DOUBLE_QUOTE_STR_RE.sub('"STR"', q)
    q = NUMBER_RE.sub("NUM", q)
    q = VAR_RE.sub("?VAR", q)
    return "fam_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:16]


def _normalize_pair(question: str, query: str) -> tuple[str, str]:
    qn = " ".join(question.strip().lower().split())
    sn = " ".join(query.strip().split())
    return qn, sn


def _core_rephrase(question: str) -> str:
    q = question.strip()
    q_low = q.lower()
    if q_low.startswith("how many "):
        return "What is the number of " + q[9:]
    if q_low.startswith("list all "):
        return "Provide all " + q[9:]
    if q_low.startswith("what is the "):
        return "Report the " + q[12:]
    if q_low.startswith("what are the "):
        return "Report the " + q[13:]
    if q_low.startswith("which "):
        return "Identify which " + q[6:]
    if q_low.startswith("how does "):
        return "Describe how " + q[9:]
    if q_low.startswith("compare "):
        return "Provide a comparison of " + q[8:]
    return "Provide the result for " + _lower_first(q)


def _topic_hint(topic: str) -> str:
    topic = (topic or "").strip().lower()
    if topic == "demand":
        return "using demand relations and survey-origin links"
    if topic == "shortage":
        return "using company shortage indicators by survey"
    if topic == "orders":
        return "using order-cancellation structures"
    if topic == "technology":
        return "using semiconductor technology-category data"
    if topic == "autonomous":
        return "using autonomous-driving development entries"
    if topic == "sales":
        return "using vehicle-sales observations"
    if topic == "comparison":
        return "using a cross-survey comparison view"
    return "using the Infineon survey knowledge graph"


def _generate_variants(question: str, topic: str) -> List[str]:
    q = _ensure_qmark(question)
    rephrased = _ensure_qmark(_core_rephrase(q))
    hint = _topic_hint(topic)
    variants = [
        q,
        _ensure_qmark(f"In the Infineon survey graph, {_lower_first(q)}"),
        _ensure_qmark(f"Using Infineon survey data, {_lower_first(rephrased)}"),
        _ensure_qmark(f"For the Infineon KG, {_lower_first(q)}"),
        _ensure_qmark(f"From Infineon data ({hint}), {_lower_first(q)}"),
        _ensure_qmark(f"Within the Infineon benchmark, {_lower_first(rephrased)}"),
        _ensure_qmark(f"Considering Infineon graph records, {_lower_first(q)}"),
    ]

    deduped: List[str] = []
    seen = set()
    for v in variants:
        key = " ".join(v.lower().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


def _validate_queries(seed_items: Sequence[Dict], graph_path: Path) -> None:
    g = Graph()
    g.parse(str(graph_path), format="turtle")
    checked = 0
    for item in seed_items:
        query = str(item.get("query", "")).strip()
        if not query:
            raise RuntimeError(f"Empty query for seed item: {item.get('id')}")
        try:
            rows = list(g.query(PREFIX + query))
        except Exception as exc:
            raise RuntimeError(
                f"Seed query execution failed for {item.get('id')}: {exc}"
            ) from exc
        if not rows:
            raise RuntimeError(
                f"Seed query returned empty results for {item.get('id')}"
            )
        checked += 1
    print(f"Validated {checked} seed queries against {graph_path}")


def build_dataset(
    seed_path: Path,
    out_path: Path,
    target_total: int,
    variants_per_question: int,
    validate: bool,
    graph_path: Path,
) -> None:
    with open(seed_path, "r", encoding="utf-8") as f:
        seed = json.load(f)

    if not isinstance(seed, list) or not seed:
        raise RuntimeError(f"Seed dataset is empty or invalid: {seed_path}")

    if validate:
        _validate_queries(seed, graph_path=graph_path)

    families: Dict[str, List[Dict]] = defaultdict(list)
    seed_label_counts = Counter()
    for idx, item in enumerate(seed, start=1):
        query = str(item.get("query", "")).strip()
        family = str(item.get("family", "")).strip() or _query_family_signature(query)
        normalized = dict(item)
        normalized["id"] = str(item.get("id", f"Q{idx}")).strip()
        normalized["topic"] = str(item.get("topic", "unknown")).strip().lower()
        normalized["ambiguity_label"] = str(item.get("ambiguity_label", "unknown")).strip().lower()
        normalized["family"] = family
        families[family].append(normalized)
        seed_label_counts[normalized["ambiguity_label"]] += 1

    # Build variant pools per family. This keeps family sizes balanced and avoids
    # over-expanding the few repeated-family templates from the seed set.
    by_family: Dict[str, Dict[str, object]] = {}

    for family, members in sorted(families.items(), key=lambda x: x[0]):
        labels = {str(m.get("ambiguity_label", "unknown")).lower() for m in members}
        if len(labels) != 1:
            raise RuntimeError(f"Mixed ambiguity labels in family {family}: {labels}")
        ambiguity = list(labels)[0]
        topic = str(members[0].get("topic", "unknown")).lower()

        candidates: List[Dict[str, str]] = []
        seen_local = set()
        for m_idx, member in enumerate(members, start=1):
            base_id = str(member.get("id", f"{family}_{m_idx}"))
            base_q = str(member.get("question", "")).strip()
            base_query = str(member.get("query", "")).strip()
            for v_idx, qv in enumerate(_generate_variants(base_q, topic=topic), start=1):
                row = {
                    "id": f"{base_id}_V{v_idx}",
                    "question": qv,
                    "query": base_query,
                    "ambiguity_label": ambiguity,
                    "topic": topic,
                    "family": family,
                    "seed_id": base_id,
                    "variant_index": v_idx,
                }
                pair_key = _normalize_pair(row["question"], row["query"])
                if pair_key in seen_local:
                    continue
                seen_local.add(pair_key)
                candidates.append(row)

        if not candidates:
            raise RuntimeError(f"No variants generated for family={family}")
        by_family[family] = {
            "label": ambiguity,
            "rows": candidates,
        }

    # Preserve label ratio from the seed dataset.
    labels = ["low", "mid", "high"]
    target_by_label: Dict[str, int] = {}
    for lab in labels:
        target_by_label[lab] = int(round(target_total * (seed_label_counts[lab] / len(seed))))
    # Fix rounding drift.
    while sum(target_by_label.values()) < target_total:
        lab = max(labels, key=lambda x: seed_label_counts[x])
        target_by_label[lab] += 1
    while sum(target_by_label.values()) > target_total:
        lab = max(labels, key=lambda x: target_by_label[x])
        target_by_label[lab] -= 1

    families_by_label: Dict[str, List[str]] = defaultdict(list)
    for fam, info in by_family.items():
        families_by_label[str(info["label"])].append(fam)
    for lab in labels:
        families_by_label[lab].sort()
        if target_by_label[lab] > 0 and not families_by_label[lab]:
            raise RuntimeError(f"Cannot satisfy target for label={lab}; no families available.")

    out: List[Dict[str, str]] = []
    seen_pairs = set()
    fam_ptr = {fam: 0 for fam in by_family}
    lab_ptr = {lab: 0 for lab in labels}
    remaining = dict(target_by_label)

    while len(out) < target_total and any(v > 0 for v in remaining.values()):
        progressed = False
        # Cycle by label to enforce the requested ambiguity distribution.
        for lab in labels:
            if remaining[lab] <= 0:
                continue
            fams = families_by_label[lab]
            if not fams:
                continue

            selected = None
            for step in range(len(fams)):
                idx = (lab_ptr[lab] + step) % len(fams)
                fam = fams[idx]
                rows = by_family[fam]["rows"]  # type: ignore[index]
                ptr = fam_ptr[fam]
                while ptr < len(rows):
                    cand = rows[ptr]
                    pair_key = _normalize_pair(cand["question"], cand["query"])
                    if pair_key not in seen_pairs:
                        selected = (fam, idx, ptr, cand, pair_key)
                        break
                    ptr += 1
                fam_ptr[fam] = ptr
                if selected is not None:
                    break

            if selected is None:
                continue

            fam, idx, ptr, cand, pair_key = selected
            seen_pairs.add(pair_key)
            out.append(cand)
            remaining[lab] -= 1
            fam_ptr[fam] = ptr + 1
            lab_ptr[lab] = (idx + 1) % len(fams)
            progressed = True

            if len(out) >= target_total:
                break

        if not progressed:
            break

    if len(out) < target_total:
        raise RuntimeError(
            f"Could not reach target_total={target_total}. "
            f"Generated only {len(out)} rows. "
            f"Remaining targets by label: {remaining}. "
            "Increase --variants-per-question."
        )

    # Ensure deterministic ID order and contiguous ids by ambiguity label.
    counters = Counter()
    normalized: List[Dict[str, str]] = []
    for item in out:
        lab = str(item.get("ambiguity_label", "unknown")).lower()
        counters[lab] += 1
        fixed = dict(item)
        fixed["id"] = f"{lab.upper()}{counters[lab]}"
        normalized.append(fixed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
        f.write("\n")

    counts = Counter(str(x.get("ambiguity_label", "unknown")).lower() for x in normalized)
    fam_count = len({x.get("family", "") for x in normalized})
    print(f"Saved {len(normalized)} questions to {out_path}")
    print(f"Label distribution: {dict(counts)}")
    print(f"Unique query families: {fam_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a 500-question Infineon benchmark with stable families."
    )
    parser.add_argument("--seed", default=str(DEFAULT_SEED))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument(
        "--variants-per-question",
        type=int,
        default=6,
        help="Maximum paraphrase variants generated per seed question.",
    )
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip query execution validation against the graph.",
    )
    args = parser.parse_args()

    build_dataset(
        seed_path=Path(args.seed),
        out_path=Path(args.out),
        target_total=max(1, int(args.target)),
        variants_per_question=max(1, int(args.variants_per_question)),
        validate=not args.no_validate,
        graph_path=Path(args.graph),
    )


if __name__ == "__main__":
    main()
