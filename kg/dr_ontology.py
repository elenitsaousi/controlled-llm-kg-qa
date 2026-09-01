from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS


PROJECT_DR_ONTOLOGY_PATH = Path(__file__).resolve().parents[1] / "data" / "digital_reference" / "DigitalReference.ttl"
DEFAULT_DR_ONTOLOGY_PATH = PROJECT_DR_ONTOLOGY_PATH
FALLBACK_DR_ONTOLOGY_PATHS = (
    Path.home() / "Downloads" / "dr" / "DigitalReference.ttl",
    Path.home() / "Downloads" / "DigitalReference.ttl",
)
DC_DESCRIPTION = URIRef("http://purl.org/dc/elements/1.1/description")

DEFINITION_PATTERNS = (
    re.compile(r"^\s*what\s+(?:is|are)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what\s+does\s+(.+?)\s+mean\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*how\s+(?:is|are)\s+(.+?)\s+defined\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*how\s+(?:do|would)\s+you\s+define\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:what\s+is\s+)?(?:the\s+)?definition\s+of\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*define\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*explain\s+(.+?)\s*\??\s*$", re.IGNORECASE),
)

COMPARISON_PATTERNS = (
    re.compile(r"^\s*(.+?)\s+(?:vs\.?|versus)\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:what\s+is\s+)?(?:the\s+)?difference\s+between\s+(.+?)\s+and\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)\s*\??\s*$", re.IGNORECASE),
)

DEFINITION_INTENT_PATTERNS = (
    re.compile(r"\b(defin(e|ed|ition)|meaning|mean|explain|describe)\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+about\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(is|are)\b", re.IGNORECASE),
)

ANALYTIC_HINTS = {
    "average",
    "between",
    "by",
    "change",
    "count",
    "decrease",
    "evolve",
    "for each",
    "group",
    "has",
    "highest",
    "how many",
    "how much",
    "increase",
    "list",
    "lowest",
    "month",
    "quarter",
    "show",
    "sum",
    "total",
    "trend",
    "year",
}

DEFINITION_LIST_SEPARATORS = re.compile(
    r"\s*(?:,|;|/|\band\b|\bor\b|\bversus\b|\bvs\.?\b)\s*",
    re.IGNORECASE,
)

GENERIC_INVENTORY_HINTS = {
    "available",
    "covered",
    "coverage",
    "do you have",
    "how many",
    "list",
    "show",
    "what can",
    "which topics",
}

GRAPH_COMPARISON_HINTS = {
    "average",
    "breakdown",
    "count",
    "group",
    "highest",
    "how many",
    "list",
    "lowest",
    "show",
    "sum",
    "total",
    "trend",
}

GRAPH_QUERY_HINTS = GRAPH_COMPARISON_HINTS | {
    "current",
    "develop",
    "developing",
    "development",
    "expected",
    "latest",
    "last",
    "monitor",
    "names of",
    "labels",
    "past",
    "risk",
    "should",
    "split",
    "upcoming",
}

COVERAGE_INTENT_HINTS = {
    "composed of",
    "consist of",
    "consists of",
    "contain",
    "contains",
    "cover",
    "covered",
    "covers",
    "include",
    "included",
    "includes",
    "made of",
    "part of",
    "parts of",
}

TARGET_STOP_TERMS = {
    "class",
    "classes",
    "concept",
    "definition",
    "meaning",
    "properties",
    "property",
    "relation",
    "relationship",
    "term",
}


@dataclass
class DROntologyTerm:
    label: str
    uri: str
    kind: str
    definition: str = ""
    aliases: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    ranges: List[str] = field(default_factory=list)


def _normalize_alias(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", unquote(str(text or "")).lower()))


def _clean_definition_target(text: str) -> str:
    target = unquote(str(text or "")).strip(" \t\r\n.?!\"'`“”‘’")
    target = re.sub(r"\s+", " ", target).strip()
    target = re.sub(
        r"^(?:the\s+)?(?:definition|meaning)\s+of\s+",
        "",
        target,
        flags=re.IGNORECASE,
    ).strip()
    target = re.sub(
        r"^(?:the\s+)?(?:concept|term|class|property|relationship|relation|object\s+property|data\s+property|datatype\s+property)\s+",
        "",
        target,
        flags=re.IGNORECASE,
    ).strip()
    target = re.sub(
        r"\s+(?:relationship|relation|object\s+property|data\s+property|datatype\s+property)\s*$",
        "",
        target,
        flags=re.IGNORECASE,
    ).strip(" \t\r\n.?!\"'`“”‘’")
    return target


def _local_name(value: object) -> str:
    text = unquote(str(value or "")).strip()
    text = text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _first_literal(graph: Graph, subject: URIRef, predicates: Iterable[URIRef]) -> str:
    for predicate in predicates:
        values = [
            str(value).strip()
            for value in graph.objects(subject, predicate)
            if isinstance(value, Literal) and str(value).strip()
        ]
        if values:
            return values[0]
    return ""


def _all_literals(graph: Graph, subject: URIRef, predicates: Iterable[URIRef]) -> List[str]:
    values: List[str] = []
    for predicate in predicates:
        for value in graph.objects(subject, predicate):
            if isinstance(value, Literal) and str(value).strip():
                values.append(str(value).strip())
    return values


def _kind(graph: Graph, subject: URIRef) -> str:
    types = set(graph.objects(subject, RDF.type))
    if OWL.Ontology in types:
        return "ontology"
    if OWL.Class in types or RDFS.Class in types:
        return "class"
    if OWL.ObjectProperty in types:
        return "object property"
    if OWL.DatatypeProperty in types:
        return "datatype property"
    if OWL.AnnotationProperty in types:
        return "annotation property"
    return "resource"


def _dr_path() -> Optional[Path]:
    configured = (
        os.getenv("TRUE_DEMAND_DR_ONTOLOGY_PATH", "").strip()
        or os.getenv("DR_ONTOLOGY_PATH", "").strip()
    )
    path = Path(configured).expanduser() if configured else DEFAULT_DR_ONTOLOGY_PATH
    if path.exists():
        return path
    if not configured:
        for fallback in FALLBACK_DR_ONTOLOGY_PATHS:
            if fallback.exists():
                return fallback
    return None


@lru_cache(maxsize=4)
def _load_dr_terms(path_text: str) -> Dict[str, DROntologyTerm]:
    graph = Graph()
    path = Path(path_text).expanduser()
    if path.exists():
        graph.parse(path.resolve().as_uri(), format="turtle")
    else:
        graph.parse(path_text, format="turtle")

    by_uri: Dict[str, DROntologyTerm] = {}
    for subject in set(graph.subjects()):
        if not isinstance(subject, URIRef):
            continue
        kind = _kind(graph, subject)
        if kind == "resource":
            continue
        labels = _all_literals(graph, subject, (RDFS.label, SKOS.prefLabel, SKOS.altLabel))
        label = labels[0] if labels else _local_name(subject)
        definition = _first_literal(graph, subject, (SKOS.definition, RDFS.comment, SKOS.scopeNote, DC_DESCRIPTION))
        by_uri[str(subject)] = DROntologyTerm(
            label=label,
            uri=str(subject),
            kind=kind,
            definition=definition,
            aliases=[label, _local_name(subject), *labels],
        )

    def label_for(uri: URIRef) -> str:
        term = by_uri.get(str(uri))
        return term.label if term else _local_name(uri)

    for uri, term in by_uri.items():
        subject = URIRef(uri)
        term.parents = sorted(
            {
                label_for(parent)
                for parent in graph.objects(subject, RDFS.subClassOf)
                if isinstance(parent, URIRef)
            }
        )
        term.domains = sorted(
            {
                label_for(domain)
                for domain in graph.objects(subject, RDFS.domain)
                if isinstance(domain, URIRef)
            }
        )
        term.ranges = sorted(
            {
                label_for(range_)
                for range_ in graph.objects(subject, RDFS.range)
                if isinstance(range_, URIRef)
            }
        )

    by_alias: Dict[str, DROntologyTerm] = {}
    for term in by_uri.values():
        for alias in term.aliases:
            key = _normalize_alias(alias)
            if not key:
                continue
            existing = by_alias.get(key)
            if existing is None or (not existing.definition and term.definition):
                by_alias[key] = term
    return by_alias


def _definition_target(question: str) -> Optional[str]:
    raw = str(question or "").strip()
    for pattern in DEFINITION_PATTERNS:
        match = pattern.match(raw)
        if not match:
            continue
        target = _clean_definition_target(match.group(1))
        return target
    return None


def _definition_targets(question: str) -> List[str]:
    target = _definition_target(question)
    if target is None:
        return []
    strong_definition_intent = bool(
        re.search(r"\b(defin(e|ed|ition)|meaning|mean)\b", str(question or ""), re.IGNORECASE)
    )
    if _looks_like_graph_query(question) and not strong_definition_intent:
        return []

    lowered_question = str(question or "").lower()
    lowered_target = target.lower()
    if (
        any(hint in lowered_question for hint in GENERIC_INVENTORY_HINTS)
        and any(word in lowered_target for word in ("class", "classes", "property", "properties", "topic", "topics"))
        and not any(mark in target for mark in (",", ";", "/", " and ", " or "))
    ):
        return []

    parts = DEFINITION_LIST_SEPARATORS.split(target)
    cleaned = [_clean_definition_target(part) for part in parts]
    cleaned = [part for part in cleaned if part and _normalize_alias(part) not in TARGET_STOP_TERMS]
    if not cleaned:
        cleaned = [_clean_definition_target(target)]

    deduped: List[str] = []
    seen = set()
    for part in cleaned:
        key = _normalize_alias(part)
        if key and key not in seen:
            deduped.append(part)
            seen.add(key)
    return deduped


def _looks_like_graph_comparison(question: str) -> bool:
    q_norm = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if any(re.search(rf"\b{re.escape(hint)}\b", q_norm) for hint in GRAPH_COMPARISON_HINTS):
        return True
    return bool(re.search(r"\bby\s+(region|month|quarter|year|survey|technology|vehicle|component|category)\b", q_norm))


def _looks_like_graph_query(question: str) -> bool:
    q_norm = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if any(re.search(rf"\b{re.escape(hint)}\b", q_norm) for hint in GRAPH_QUERY_HINTS):
        return True
    return bool(re.search(r"\bby\s+(region|month|quarter|year|survey|technology|vehicle|component|category)\b", q_norm))


def _looks_like_coverage_request(question: str) -> bool:
    q_norm = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if not any(hint in q_norm for hint in COVERAGE_INTENT_HINTS):
        return False
    return bool(re.search(r"\b(true demand|digital reference|dr ontology|ontology|kg|knowledge graph|source|sources|graph)\b", q_norm))


def _has_definition_intent(question: str) -> bool:
    q_norm = re.sub(r"\s+", " ", str(question or "").lower()).strip()
    if not q_norm or _looks_like_coverage_request(q_norm):
        return False
    strong_definition_intent = bool(re.search(r"\b(defin(e|ed|ition)|meaning|mean)\b", q_norm))
    if _looks_like_graph_query(q_norm) and not strong_definition_intent:
        return False
    if (
        any(hint in q_norm for hint in GENERIC_INVENTORY_HINTS)
        and re.search(r"\b(classes?|properties?|topics?|capabilities|questions?)\b", q_norm)
    ):
        return False
    return any(pattern.search(q_norm) for pattern in DEFINITION_INTENT_PATTERNS)


def _term_label_by_uri(terms: Dict[str, DROntologyTerm]) -> Dict[str, DROntologyTerm]:
    unique: Dict[str, DROntologyTerm] = {}
    for term in terms.values():
        unique[term.uri] = term
    return unique


def _known_targets_from_question(question: str, terms: Dict[str, DROntologyTerm]) -> List[str]:
    if not _has_definition_intent(question):
        return []

    raw = str(question or "")
    candidates: List[Tuple[int, int, str]] = []

    for label in PROJECT_GLOSSARY:
        if len(label) < 4 or _normalize_alias(label) in TARGET_STOP_TERMS:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(label).replace(r'\ ', r'\s+')}(?![A-Za-z0-9])", re.IGNORECASE)
        for match in pattern.finditer(raw):
            candidates.append((match.start(), match.end(), label))

    for term in _term_label_by_uri(terms).values():
        aliases = [term.label, *term.aliases]
        for alias in dict.fromkeys(alias for alias in aliases if len(str(alias).strip()) >= 4):
            alias_text = str(alias).strip()
            if _normalize_alias(alias_text) in TARGET_STOP_TERMS:
                continue
            if len(alias_text.split()) == 1 and len(alias_text) < 5:
                continue
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(alias_text).replace(r'\ ', r'\s+')}(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            for match in pattern.finditer(raw):
                candidates.append((match.start(), match.end(), term.label))

    candidates.sort(key=lambda item: (item[1] - item[0], -item[0]), reverse=True)
    accepted: List[Tuple[int, int, str]] = []
    for start, end, label in candidates:
        if any(not (end <= taken_start or start >= taken_end) for taken_start, taken_end, _ in accepted):
            continue
        accepted.append((start, end, label))

    accepted.sort(key=lambda item: item[0])
    deduped: List[str] = []
    seen = set()
    for _start, _end, label in accepted:
        key = _normalize_alias(label)
        if key and key not in seen:
            deduped.append(label)
            seen.add(key)
    return deduped


def _comparison_targets(question: str) -> List[str]:
    if _looks_like_graph_comparison(question):
        return []
    raw = str(question or "").strip()
    for pattern in COMPARISON_PATTERNS:
        match = pattern.match(raw)
        if not match:
            continue
        targets = [_clean_definition_target(match.group(1)), _clean_definition_target(match.group(2))]
        deduped: List[str] = []
        seen = set()
        for target in targets:
            key = _normalize_alias(target)
            if key and key not in seen:
                deduped.append(target)
                seen.add(key)
        return deduped if len(deduped) >= 2 else []
    return []


def _best_term(target: str, terms: Dict[str, DROntologyTerm]) -> Optional[DROntologyTerm]:
    key = _normalize_alias(target)
    if not key:
        return None
    if key in terms:
        return terms[key]

    candidates: List[Tuple[float, int, DROntologyTerm]] = []
    for alias, term in terms.items():
        if len(key) < 4:
            continue
        if alias.startswith(key) and "property" in term.kind.lower():
            property_bonus = 0.15 if "property" in term.kind.lower() else 0.0
            candidates.append((0.72 + property_bonus, len(alias), term))
            continue
        if key in alias:
            candidates.append((len(key) / max(len(alias), 1), len(alias), term))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    score, _, term = candidates[0]
    return term if score >= 0.55 else None


PROJECT_GLOSSARY: Dict[str, Tuple[str, str]] = {
    "true demand": (
        "project concept",
        "True Demand describes demand information intended to approximate real market need rather than inflated or safety-stock-driven forecasts. In this system it is represented through survey-based current-demand, future-demand, shortage, inventory, vehicle-sales, and related semiconductor supply-chain signals.",
    ),
    "future demand": (
        "True Demand KG concept",
        "Future Demand describes survey-reported expected demand signals in the True Demand knowledge graph. It can be analyzed by supported dimensions such as region, quarter, survey origin, vehicle type, component share, or technology category when the corresponding graph path exists.",
    ),
    "expected future demand": (
        "True Demand KG concept",
        "Expected future demand is treated as Future Demand in this system: survey-reported expected demand signals that can be analyzed by supported dimensions such as region, quarter, survey origin, vehicle type, component share, or technology category.",
    ),
    "current demand": (
        "True Demand KG concept",
        "Current Demand describes survey-reported present demand signals in the True Demand knowledge graph. It is used for graph-supported breakdowns such as current demand by region, survey origin, vehicle type, or related demand dimensions.",
    ),
    "technology node": (
        "True Demand KG concept",
        "Technology Node represents a semiconductor technology category or node used to group demand, inventory, shortage, and related survey responses in the True Demand knowledge graph.",
    ),
    "oem": (
        "supply-chain actor",
        "OEM means Original Equipment Manufacturer. In this system, OEM is used as a survey/source group for demand and supply-chain signals reported from original equipment manufacturers.",
    ),
    "tier1": (
        "supply-chain actor",
        "Tier1 refers to first-tier suppliers that provide systems, modules, or components directly to OEMs. In this system, Tier1 is used as a survey/source group for demand, inventory, and related supply-chain signals.",
    ),
    "tier 1": (
        "supply-chain actor",
        "Tier1 refers to first-tier suppliers that provide systems, modules, or components directly to OEMs. In this system, Tier1 is used as a survey/source group for demand, inventory, and related supply-chain signals.",
    ),
    "semis": (
        "supply-chain actor",
        "Semis is shorthand for the semiconductor survey/source group in this system. It refers to semiconductor-side demand and supply-chain signals in the True Demand knowledge graph.",
    ),
    "lobe": (
        "Digital Reference modelling concept",
        "A lobe is a high-level Digital Reference domain area used to organize related concepts and relationships. It helps separate vocabulary by business or technical scope while still allowing links across domains when needed.",
    ),
    "single lobe": (
        "Digital Reference modelling concept",
        "Single-lobe modelling means that the requested concept, property, or use case can be described within one Digital Reference lobe. The relevant classes and relationships mainly stay inside the same domain area.",
    ),
    "cross lobe": (
        "Digital Reference modelling concept",
        "Cross-lobe modelling means that the requested concept, property, or use case connects concepts from more than one Digital Reference lobe. It is used when the business meaning depends on relationships across domain areas rather than a single isolated vocabulary branch.",
    ),
}

PROJECT_GLOSSARY_BY_ALIAS: Dict[str, Tuple[str, str, str]] = {
    _normalize_alias(label): (
        {"oem": "OEM", "tier1": "Tier1", "tier 1": "Tier1", "semis": "Semis"}.get(label, label.title()),
        kind,
        definition,
    )
    for label, (kind, definition) in PROJECT_GLOSSARY.items()
}


def _project_glossary_match(target: str) -> Optional[Tuple[str, str, str]]:
    key = _normalize_alias(target)
    return PROJECT_GLOSSARY_BY_ALIAS.get(key)


def _term_answer(term: DROntologyTerm) -> str:
    answer_parts = [
        term.definition
        or f"The Digital Reference ontology contains {term.label} as a {term.kind}."
    ]
    if term.parents:
        answer_parts.append(f"In the Digital Reference ontology, it is a subclass of {_join(term.parents)}.")
    if term.domains or term.ranges:
        answer_parts.append(
            "Its declared domain is "
            + (_join(term.domains) or "not specified")
            + " and its declared range is "
            + (_join(term.ranges) or "not specified")
            + "."
        )
    return " ".join(answer_parts)


def _term_to_dict(term: DROntologyTerm) -> Dict[str, object]:
    return {
        "label": term.label,
        "uri": term.uri,
        "kind": term.kind,
        "definition": term.definition,
        "aliases": list(dict.fromkeys(term.aliases)),
        "parents": list(term.parents),
        "domains": list(term.domains),
        "ranges": list(term.ranges),
    }


def dr_ontology_terms(path_text: str = "") -> List[Dict[str, object]]:
    """Return all discoverable DR ontology terms as UI-friendly dictionaries."""
    path = Path(path_text).expanduser() if str(path_text or "").strip() else _dr_path()
    if path is None or not path.exists():
        return []
    terms_by_alias = _load_dr_terms(str(path))
    unique: Dict[str, DROntologyTerm] = {}
    for term in terms_by_alias.values():
        unique[term.uri] = term
    return [
        _term_to_dict(term)
        for term in sorted(
            unique.values(),
            key=lambda item: (item.kind, item.label.lower(), item.uri),
        )
    ]


def dr_ontology_counts(path_text: str = "") -> Dict[str, int]:
    path = Path(path_text).expanduser() if str(path_text or "").strip() else _dr_path()
    searchable_entries = 0
    if path is not None and path.exists():
        searchable_entries = len(_load_dr_terms(str(path)))
    counts: Dict[str, int] = {}
    for term in dr_ontology_terms(path_text):
        kind = str(term.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    counts["total"] = sum(value for key, value in counts.items() if key != "total")
    counts["searchable_entries"] = searchable_entries
    return counts


def search_dr_ontology_terms(
    search: str = "",
    kinds: Optional[Iterable[str]] = None,
    limit: int = 50,
    path_text: str = "",
) -> List[Dict[str, object]]:
    """Search DR terms by label, URI local name, aliases, and definition text."""
    selected_kinds = {str(kind).lower() for kind in (kinds or []) if str(kind).strip()}
    query = str(search or "").strip()
    query_key = _normalize_alias(query)
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    rows: List[Tuple[float, str, Dict[str, object]]] = []

    for term in dr_ontology_terms(path_text):
        kind = str(term.get("kind") or "").lower()
        if selected_kinds and kind not in selected_kinds:
            continue
        label = str(term.get("label") or "")
        aliases = [str(alias) for alias in term.get("aliases") or []]
        definition = str(term.get("definition") or "")
        haystack = " ".join([label, *aliases, definition, str(term.get("uri") or "")])
        haystack_key = _normalize_alias(haystack)
        haystack_tokens = set(re.findall(r"[a-z0-9]+", haystack.lower()))

        if not query_key:
            score = 0.1
        elif query_key == _normalize_alias(label):
            score = 5.0
        elif _normalize_alias(label).startswith(query_key):
            score = 4.0
        elif query_key in haystack_key:
            score = 3.0
        else:
            overlap = len(query_tokens & haystack_tokens)
            if overlap <= 0:
                continue
            score = 1.0 + overlap / max(len(query_tokens), 1)
        rows.append((score, label.lower(), term))

    rows.sort(key=lambda item: (-item[0], item[1]))
    return [term for _score, _label, term in rows[: max(1, int(limit))]]


def _join(values: Iterable[str], limit: int = 5) -> str:
    unique = list(dict.fromkeys(str(value) for value in values if str(value).strip()))
    return ", ".join(unique[:limit])


def _resolve_definition_targets(
    targets: Iterable[str],
    path: Optional[Path],
    terms: Optional[Dict[str, DROntologyTerm]] = None,
) -> Tuple[List[Dict[str, str]], List[str]]:
    found: List[Dict[str, str]] = []
    missing: List[str] = []
    for target in targets:
        term = _best_term(target, terms or {}) if terms else None
        if term is not None:
            found.append(
                {
                    "label": term.label,
                    "kind": term.kind,
                    "answer": _term_answer(term),
                    "uri": term.uri,
                    "source": "digital_reference_ontology",
                }
            )
            continue
        glossary_match = _project_glossary_match(target)
        if glossary_match is None:
            missing.append(target)
            continue
        label, kind, definition = glossary_match
        found.append(
            {
                "label": label,
                "kind": kind,
                "answer": definition,
                "uri": "",
                "source": "true_demand_project_glossary",
            }
        )
    return found, missing


def _comparison_answer(found: List[Dict[str, str]]) -> str:
    labels = [item["label"] for item in found]
    lines = [
        f"This is a terminology comparison between **{labels[0]}** and **{labels[1]}**.",
        "",
    ]
    for item in found:
        lines.append(f"- **{item['label']}** ({item['kind']}): {item['answer']}")
    if len(found) == 2:
        lines.extend(
            [
                "",
                f"In short, **{labels[0]}** and **{labels[1]}** are distinct terms in the system vocabulary. "
                f"Use **{labels[0]}** or **{labels[1]}** according to the specific meaning described above.",
            ]
        )
    return "\n".join(lines)


def route_dr_ontology_definition(question: str) -> Optional[Dict[str, object]]:
    if _looks_like_coverage_request(question):
        return None

    comparison_targets = _comparison_targets(question)
    if comparison_targets:
        path = _dr_path()
        terms = _load_dr_terms(str(path)) if path is not None else None
        found, missing = _resolve_definition_targets(comparison_targets, path, terms)
        if len(found) >= 2 and not missing:
            return {
                "route": "definition_comparison",
                "answer": _comparison_answer(found),
                "matched_term": _join([item["label"] for item in found], limit=8),
                "confidence": "High",
                "reason": "The question compares known ontology or project glossary concepts.",
                "source": "digital_reference_ontology",
                "term_kind": "comparison",
                "term_uri": _join([item["uri"] for item in found if item["uri"]], limit=8),
                "ontology_path": str(path or ""),
            }

    path = _dr_path()
    terms = _load_dr_terms(str(path)) if path is not None else None
    targets = _definition_targets(question)
    if not targets:
        targets = _known_targets_from_question(question, terms or {})
    if not targets:
        return None

    if path is None:
        found, missing = _resolve_definition_targets(targets, path)
        if not found:
            return {
                "route": "definition",
                "answer": "I could not find reliable Digital Reference or True Demand glossary definitions for: "
                + _join(missing, limit=8)
                + ".",
                "matched_term": "",
                "confidence": "Low",
                "reason": "The question asks for definitions, but no deterministic ontology or glossary match was found.",
                "source": "digital_reference_ontology",
                "term_kind": "",
                "term_uri": "",
                "ontology_path": "",
            }
        if len(found) == 1 and not missing:
            item = found[0]
            return {
                "route": "definition",
                "answer": item["answer"],
                "matched_term": item["label"],
                "confidence": "High",
                "reason": "The question asks for a deterministic True Demand glossary definition.",
                "source": item["source"],
                "term_kind": item["kind"],
                "term_uri": item["uri"],
                "ontology_path": "",
            }
        answer = "\n".join(f"- **{item['label']}** ({item['kind']}): {item['answer']}" for item in found)
        if missing:
            answer += "\n\nI could not find reliable definitions for: " + _join(missing, limit=8) + "."
        return {
            "route": "definition",
            "answer": answer,
            "matched_term": _join([item["label"] for item in found], limit=8),
            "confidence": "High",
            "reason": "The question asks for multiple deterministic True Demand / Digital Reference definitions.",
            "source": "true_demand_project_glossary",
            "term_kind": "multiple",
            "term_uri": "",
            "ontology_path": "",
        }

    found, missing = _resolve_definition_targets(targets, path, terms)

    if not found:
        return {
            "route": "definition",
            "answer": "I could not find reliable Digital Reference or True Demand glossary definitions for: "
            + _join(missing, limit=8)
            + ".",
            "matched_term": "",
            "confidence": "Low",
            "reason": "The question asks for definitions, but no deterministic ontology or glossary match was found.",
            "source": "digital_reference_ontology",
            "term_kind": "",
            "term_uri": "",
            "ontology_path": str(path),
        }

    if len(found) == 1 and not missing:
        item = found[0]
        return {
            "route": "definition",
            "answer": item["answer"],
            "matched_term": item["label"],
            "confidence": "High",
            "reason": "The question asks for a Digital Reference ontology definition.",
            "source": item["source"],
            "term_kind": item["kind"],
            "term_uri": item["uri"],
            "ontology_path": str(path),
        }

    answer = "\n".join(f"- **{item['label']}** ({item['kind']}): {item['answer']}" for item in found)
    if missing:
        answer += "\n\nI could not find reliable definitions for: " + _join(missing, limit=8) + "."

    return {
        "route": "definition",
        "answer": answer,
        "matched_term": _join([item["label"] for item in found], limit=8),
        "confidence": "High",
        "reason": "The question asks for multiple Digital Reference / True Demand definitions.",
        "source": "digital_reference_ontology",
        "term_kind": "multiple",
        "term_uri": _join([item["uri"] for item in found if item["uri"]], limit=8),
        "ontology_path": str(path),
    }
