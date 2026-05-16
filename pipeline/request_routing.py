from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote


CURATED_TERMS = {
    "bev": {
        "label": "BEV",
        "definition": "BEV means battery electric vehicle: a vehicle powered only by an electric battery and motor.",
    },
    "batteryelectricvehicle": {
        "label": "battery electric vehicle",
        "definition": "A battery electric vehicle is a vehicle powered only by an electric battery and motor. It is commonly abbreviated as BEV.",
    },
    "ice": {
        "label": "ICE",
        "definition": "ICE means internal combustion engine.",
    },
    "oem": {
        "label": "OEM",
        "definition": "OEM means original equipment manufacturer.",
    },
    "tier1": {
        "label": "Tier1",
        "definition": "Tier1 refers to a first-tier supplier that provides components directly to an OEM.",
    },
    "bl1": {
        "label": "BL1",
        "definition": "BL1 is baseline scenario 1 in the survey data.",
    },
    "bl2": {
        "label": "BL2",
        "definition": "BL2 is baseline scenario 2 in the survey data.",
    },
    "adas": {
        "label": "ADAS",
        "definition": "ADAS means advanced driver-assistance systems.",
    },
    "sae": {
        "label": "SAE",
        "definition": "SAE refers to the driving-automation level scale used for autonomous-driving data.",
    },
}

KG_HINTS = {
    "demand",
    "sales",
    "vehicle",
    "vehicles",
    "technology",
    "technologies",
    "region",
    "regions",
    "quarter",
    "quarters",
    "month",
    "monthly",
    "year",
    "yearly",
    "baseline",
    "shortage",
    "shortages",
    "inventory",
    "cancellation",
    "component",
    "components",
    "autonomous",
    "percentage",
    "percentages",
    "survey",
    "forecast",
    "actual",
}

OUT_OF_DOMAIN_HINTS = {
    "weather",
    "temperature",
    "rain",
    "football",
    "soccer",
    "president",
    "stock price",
    "recipe",
}

DEFINITION_PATTERNS = (
    re.compile(r"^\s*what\s+is\s+(?:the\s+)?(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what\s+does\s+(.+?)\s+mean\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*define\s+(.+?)\s*\??\s*$", re.IGNORECASE),
)

UNDERSPECIFIED_PATTERNS = (
    re.compile(r"^\s*what\s+about\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*tell\s+me\s+about\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*show\s+me\s+(.+?)\s*\??\s*$", re.IGNORECASE),
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _normalize_alias(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _display_name(raw: str) -> str:
    text = unquote(str(raw or "")).strip()
    text = text.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _generic_definition(label: str, kind: str) -> str:
    if kind == "class":
        return f"{label} is a class used in the Infineon knowledge graph."
    if kind == "predicate":
        return f"{label} is a relationship used in the Infineon knowledge graph."
    if kind == "property":
        return f"{label} is a property used in the Infineon knowledge graph."
    return f"{label} is a named value used in the Infineon knowledge graph."


def build_domain_glossary(schema: Any, alias_index: Optional[Any]) -> Dict[str, Dict[str, str]]:
    glossary: Dict[str, Dict[str, str]] = dict(CURATED_TERMS)

    for kind, attr in (("class", "classes"), ("predicate", "predicates"), ("property", "properties")):
        for raw in list(getattr(schema, attr, []) or []):
            label = _display_name(str(raw))
            key = _normalize_alias(label)
            if not key or key in glossary:
                continue
            glossary[key] = {
                "label": label,
                "definition": _generic_definition(label, kind),
                "kind": kind,
            }

    if alias_index is not None:
        for key, labels in alias_index.key_to_labels.items():
            if key in glossary:
                continue
            label = alias_index.best_label(key)
            if not label:
                continue
            display = _display_name(label)
            glossary[key] = {
                "label": display,
                "definition": _generic_definition(display, "named_value"),
                "kind": "named_value",
            }

    return glossary


def _recognized_term(text: str, glossary: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    return glossary.get(_normalize_alias(text))


def _contains_domain_term(question: str, glossary: Dict[str, Dict[str, str]]) -> bool:
    q = _normalize(question)
    compact = _normalize_alias(q)
    if any(key and key in compact for key in glossary):
        return True
    words = set(re.findall(r"[a-z0-9]+", q))
    return bool(words & KG_HINTS)


def _definition_route(
    question: str,
    glossary: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, object]]:
    for pattern in DEFINITION_PATTERNS:
        match = pattern.match(question or "")
        if not match:
            continue
        term = _recognized_term(match.group(1), glossary)
        if term is None:
            return None
        return {
            "route": "definition",
            "answer": term["definition"],
            "matched_term": term["label"],
            "confidence": "High",
            "reason": "The question asks for the meaning of a recognized domain term.",
        }
    return None


def _unknown_definition_route(question: str) -> Optional[Dict[str, object]]:
    for pattern in DEFINITION_PATTERNS:
        match = pattern.match(question or "")
        if not match:
            continue
        term = match.group(1).strip(" .?!")
        return {
            "route": "unknown_definition",
            "answer": f'I do not have a graph-backed definition for "{term}".',
            "confidence": "High",
            "reason": "The question asks for a definition, but the term is not present in the Infineon KG glossary.",
        }
    return None


def _underspecified_route(
    question: str,
    glossary: Dict[str, Dict[str, str]],
) -> Optional[Dict[str, object]]:
    for pattern in UNDERSPECIFIED_PATTERNS:
        match = pattern.match(question or "")
        if not match:
            continue
        term = _recognized_term(match.group(1), glossary)
        if term is None:
            return None
        label = term["label"]
        options: List[Dict[str, str]] = [
            {
                "id": "definition",
                "label": f"What does {label} mean?",
                "rewritten_question": f"What does {label} mean?",
            },
            {
                "id": "related_data",
                "label": f"Show data related to {label}.",
                "rewritten_question": f"Show data related to {label}.",
            },
        ]
        if _normalize_alias(label) in {"bev", "ice"}:
            options.extend(
                [
                    {
                        "id": "future_demand",
                        "label": f"Show future-demand results for {label}.",
                        "rewritten_question": f"Show future-demand results for {label}.",
                    },
                    {
                        "id": "autonomous",
                        "label": f"Show autonomous-driving percentages for {label}.",
                        "rewritten_question": f"Show autonomous-driving percentages for {label}.",
                    },
                ]
            )
        return {
            "route": "clarification_needed",
            "request_clarification": {
                "needs_clarification": True,
                "reason": f"`{label}` is recognized, but the requested task is not specified.",
                "question": "What do you want to know?",
                "options": options,
            },
            "confidence": "Low",
        }
    return None


def route_request(
    question: str,
    *,
    schema: Any = None,
    alias_index: Optional[Any] = None,
) -> Dict[str, object]:
    glossary = build_domain_glossary(schema, alias_index) if schema is not None else dict(CURATED_TERMS)

    definition = _definition_route(question, glossary)
    if definition is not None:
        return definition

    q = _normalize(question)
    if any(hint in q for hint in OUT_OF_DOMAIN_HINTS) and not _contains_domain_term(question, glossary):
        return {
            "route": "out_of_domain",
            "answer": "I can answer questions about the Infineon knowledge graph, but this question is outside the available data.",
            "confidence": "High",
            "reason": "No Infineon KG concept was detected and the request appears outside the dataset scope.",
        }

    unknown_definition = _unknown_definition_route(question)
    if unknown_definition is not None:
        return unknown_definition

    underspecified = _underspecified_route(question, glossary)
    if underspecified is not None:
        return underspecified

    return {"route": "kg_query"}
