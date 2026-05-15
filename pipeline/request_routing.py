from __future__ import annotations

import re
from typing import Dict, List, Optional


DOMAIN_TERMS = {
    "bev": {
        "label": "BEV",
        "definition": "BEV means battery electric vehicle: a vehicle powered only by an electric battery and motor.",
    },
    "battery electric vehicle": {
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
    "tier 1": {
        "label": "Tier 1",
        "definition": "Tier 1 refers to a first-tier supplier that provides components directly to an OEM.",
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


def _recognized_term(text: str) -> Optional[Dict[str, str]]:
    key = _normalize(text).strip(" .?!")
    return DOMAIN_TERMS.get(key)


def _contains_domain_term(question: str) -> bool:
    q = _normalize(question)
    if any(term in q for term in DOMAIN_TERMS):
        return True
    words = set(re.findall(r"[a-z0-9]+", q))
    return bool(words & KG_HINTS)


def _definition_route(question: str) -> Optional[Dict[str, object]]:
    for pattern in DEFINITION_PATTERNS:
        match = pattern.match(question or "")
        if not match:
            continue
        term = _recognized_term(match.group(1))
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


def _underspecified_route(question: str) -> Optional[Dict[str, object]]:
    for pattern in UNDERSPECIFIED_PATTERNS:
        match = pattern.match(question or "")
        if not match:
            continue
        term = _recognized_term(match.group(1))
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


def route_request(question: str) -> Dict[str, object]:
    definition = _definition_route(question)
    if definition is not None:
        return definition

    underspecified = _underspecified_route(question)
    if underspecified is not None:
        return underspecified

    q = _normalize(question)
    if any(hint in q for hint in OUT_OF_DOMAIN_HINTS) and not _contains_domain_term(question):
        return {
            "route": "out_of_domain",
            "answer": "I can answer questions about the Infineon knowledge graph, but this question is outside the available data.",
            "confidence": "High",
            "reason": "No Infineon KG concept was detected and the request appears outside the dataset scope.",
        }

    return {"route": "kg_query"}
