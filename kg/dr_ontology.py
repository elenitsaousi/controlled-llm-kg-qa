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


DEFAULT_DR_ONTOLOGY_PATH = Path.home() / "Downloads" / "dr" / "DigitalReference.ttl"
FALLBACK_DR_ONTOLOGY_PATHS = (
    Path.home() / "Downloads" / "DigitalReference.ttl",
)
DC_DESCRIPTION = URIRef("http://purl.org/dc/elements/1.1/description")

DEFINITION_PATTERNS = (
    re.compile(r"^\s*what\s+(?:is|are)\s+(?:the\s+|a\s+|an\s+)?(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what\s+does\s+(.+?)\s+mean\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*define\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*explain\s+(.+?)\s*\??\s*$", re.IGNORECASE),
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
        target = match.group(1).strip(" .?!")
        return target
    return None


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
    "current demand": (
        "True Demand KG concept",
        "Current Demand describes survey-reported present demand signals in the True Demand knowledge graph. It is used for graph-supported breakdowns such as current demand by region, survey origin, vehicle type, or related demand dimensions.",
    ),
    "technology node": (
        "True Demand KG concept",
        "Technology Node represents a semiconductor technology category or node used to group demand, inventory, shortage, and related survey responses in the True Demand knowledge graph.",
    ),
}

PROJECT_GLOSSARY_BY_ALIAS: Dict[str, Tuple[str, str, str]] = {
    _normalize_alias(label): (label.title(), kind, definition)
    for label, (kind, definition) in PROJECT_GLOSSARY.items()
}


def _project_glossary_match(target: str) -> Optional[Tuple[str, str, str]]:
    key = _normalize_alias(target)
    return PROJECT_GLOSSARY_BY_ALIAS.get(key)


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


def route_dr_ontology_definition(question: str) -> Optional[Dict[str, object]]:
    target = _definition_target(question)
    if target is None:
        return None
    path = _dr_path()
    if path is None:
        glossary_match = _project_glossary_match(target)
        if glossary_match is None:
            return None
        label, kind, definition = glossary_match
        return {
            "route": "definition",
            "answer": definition,
            "matched_term": label,
            "confidence": "High",
            "reason": "The question asks for a deterministic True Demand glossary definition.",
            "source": "true_demand_project_glossary",
            "term_kind": kind,
            "term_uri": "",
            "ontology_path": "",
        }
    terms = _load_dr_terms(str(path))
    term = _best_term(target, terms)
    if term is None:
        glossary_match = _project_glossary_match(target)
        if glossary_match is None:
            return None
        label, kind, definition = glossary_match
        return {
            "route": "definition",
            "answer": definition,
            "matched_term": label,
            "confidence": "High",
            "reason": "The question asks for a deterministic True Demand / Digital Reference glossary definition.",
            "source": "true_demand_project_glossary",
            "term_kind": kind,
            "term_uri": "",
            "ontology_path": str(path),
        }
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

    return {
        "route": "definition",
        "answer": " ".join(answer_parts),
        "matched_term": term.label,
        "confidence": "High",
        "reason": "The question asks for a Digital Reference ontology definition.",
        "source": "digital_reference_ontology",
        "term_kind": term.kind,
        "term_uri": term.uri,
        "ontology_path": str(path),
    }
