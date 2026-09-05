from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence
from urllib.parse import unquote

from rdflib import Graph, Literal, RDF, URIRef

from kg.entity_linking import TARGET_PREDICATE_URIS


PLACEHOLDER_LABELS = {"", "-", "#", "n/a", "na", "none", "null", "unknown", "not available"}
NONWORD_RE = re.compile(r"[^a-z0-9]+")


def local_name(value: object) -> str:
    text = unquote(str(value or "")).strip()
    return text.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def is_placeholder_label(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in PLACEHOLDER_LABELS or not NONWORD_RE.sub("", text)


def _display_local_name(value: object) -> str:
    text = local_name(value).replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class EntityProfile:
    iri: str
    canonical_label: str
    types: List[str]
    aliases: List[str]
    predicates: List[str]
    quality_flags: List[str]


def build_entity_profiles(graph: Graph) -> Dict[str, EntityProfile]:
    labels: Dict[URIRef, List[str]] = defaultdict(list)
    types: Dict[URIRef, List[str]] = defaultdict(list)
    predicates: Dict[URIRef, List[str]] = defaultdict(list)

    for s, p, o in graph:
        if not isinstance(s, URIRef):
            continue
        predicates[s].append(local_name(p))
        if p == RDF.type and isinstance(o, URIRef):
            types[s].append(local_name(o))
        if isinstance(o, Literal) and str(p) in TARGET_PREDICATE_URIS:
            labels[s].append(str(o).strip())

    profiles: Dict[str, EntityProfile] = {}
    for subject in sorted(set(predicates) | set(types) | set(labels), key=str):
        raw_labels = labels.get(subject, [])
        clean_labels = [label for label in raw_labels if not is_placeholder_label(label)]
        fallback = _display_local_name(subject)
        canonical = clean_labels[0] if clean_labels else fallback
        flags: List[str] = []
        if not raw_labels:
            flags.append("missing_display_label")
        elif not clean_labels:
            flags.append("placeholder_display_label")
        if not types.get(subject):
            flags.append("missing_rdf_type")
        profiles[str(subject)] = EntityProfile(
            iri=str(subject),
            canonical_label=canonical,
            types=sorted(set(types.get(subject, []))),
            aliases=sorted({*clean_labels, fallback}),
            predicates=sorted(set(predicates.get(subject, []))),
            quality_flags=flags,
        )
    return profiles


def build_profiles_for_iris(graph: Graph, iris: Sequence[str]) -> List[EntityProfile]:
    subjects = [URIRef(iri) for iri in sorted(set(iris)) if iri]
    profiles: List[EntityProfile] = []
    for subject in subjects:
        labels: List[str] = []
        types: List[str] = []
        predicates: List[str] = []
        for _, predicate, obj in graph.triples((subject, None, None)):
            predicates.append(local_name(predicate))
            if predicate == RDF.type and isinstance(obj, URIRef):
                types.append(local_name(obj))
            if isinstance(obj, Literal) and str(predicate) in TARGET_PREDICATE_URIS:
                labels.append(str(obj).strip())
        clean_labels = [label for label in labels if not is_placeholder_label(label)]
        fallback = _display_local_name(subject)
        flags: List[str] = []
        if not labels:
            flags.append("missing_display_label")
        elif not clean_labels:
            flags.append("placeholder_display_label")
        if not types:
            flags.append("missing_rdf_type")
        profiles.append(
            EntityProfile(
                iri=str(subject),
                canonical_label=clean_labels[0] if clean_labels else fallback,
                types=sorted(set(types)),
                aliases=sorted({*clean_labels, fallback}),
                predicates=sorted(set(predicates)),
                quality_flags=flags,
            )
        )
    return profiles


def summarize_graph_quality(
    graph: Graph,
    profiles: Dict[str, EntityProfile] | None = None,
) -> Dict[str, object]:
    entity_profiles = profiles or build_entity_profiles(graph)
    by_type: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    examples: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for profile in entity_profiles.values():
        by_type.update(profile.types or ["<untyped>"])
        flag_counts.update(profile.quality_flags)
        for flag in profile.quality_flags:
            if len(examples[flag]) < 10:
                examples[flag].append(
                    {
                        "iri": profile.iri,
                        "canonical_label": profile.canonical_label,
                        "types": profile.types,
                    }
                )

    return {
        "triple_count": len(graph),
        "entity_count": len(entity_profiles),
        "entities_by_type": dict(sorted(by_type.items())),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "quality_flag_examples": dict(examples),
    }


def profile_prompt_lines(profiles: Sequence[EntityProfile], max_profiles: int = 5) -> List[str]:
    lines: List[str] = []
    for profile in list(profiles)[:max_profiles]:
        type_text = ", ".join(profile.types) if profile.types else "untyped"
        predicate_text = ", ".join(profile.predicates[:5])
        flag_text = ", ".join(profile.quality_flags) if profile.quality_flags else "none"
        lines.append(
            f"- {profile.canonical_label}: types={type_text}; "
            f"predicates={predicate_text}; quality_flags={flag_text}"
        )
    return lines


def normalize_profile_alias(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def build_profile_alias_lookup(
    profiles: Dict[str, EntityProfile],
) -> Dict[str, List[EntityProfile]]:
    lookup: Dict[str, List[EntityProfile]] = defaultdict(list)
    for profile in profiles.values():
        for alias in profile.aliases:
            key = normalize_profile_alias(alias)
            if key:
                lookup[key].append(profile)
    return dict(lookup)
