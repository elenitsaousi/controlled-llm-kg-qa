from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from kg.dr_ontology import route_dr_ontology_definition


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
    "truedemand": {
        "label": "True Demand",
        "definition": (
            "True Demand is the survey-grounded estimate of real semiconductor demand. "
            "It is used to reduce planning based on assumptions or inflated forecasts and "
            "to support more reliable supply-chain and manufacturing decisions."
        ),
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

QUERY_TASK_HINTS = {
    "average",
    "avg",
    "breakdown",
    "compare",
    "comparison",
    "count",
    "counts",
    "current",
    "each",
    "forecasted",
    "future",
    "highest",
    "largest",
    "level",
    "levels",
    "lowest",
    "number",
    "overall",
    "per",
    "records",
    "reported",
    "show",
    "sum",
    "total",
    "trend",
    "trends",
    "units",
}

OUT_OF_DOMAIN_HINTS = {
    "weather",
    "temperature",
    "rain",
    "football",
    "soccer",
    "president",
    "stock price",
    "exchange rate",
    "restaurant",
    "recipe",
}

UNSUPPORTED_RELATIVE_TIME_PATTERNS = (
    re.compile(
        r"\b(?:past|last|recent|previous)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*"
        r"(?:days?|weeks?|months?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:over|during|in)\s+the\s+(?:past|last|recent|previous)\s+"
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*"
        r"(?:days?|weeks?|months?)\b",
        re.IGNORECASE,
    ),
)

UNSUPPORTED_LIVE_TIME_PATTERNS = (
    re.compile(r"\b(?:today|right now|real[- ]?time|live|latest|up[- ]?to[- ]date)\b", re.IGNORECASE),
    re.compile(r"\b(?:currently|now)\b", re.IGNORECASE),
)

TREND_INTENT_PATTERN = re.compile(
    r"\b(?:develop(?:ing|ment)?|evolv(?:e|ing|es)|trend(?:s|ing)?|change(?:s|d)? over time|progress(?:ion)?)\b",
    re.IGNORECASE,
)

SUPPORTED_TIME_OR_BREAKDOWN_PATTERN = re.compile(
    r"\b(?:by|per|across|grouped by|broken down by|for each)\s+"
    r"(?:month|months|quarter|quarters|year|years|region|regions|technology|technology category|vehicle|vehicle type|survey|survey group)\b",
    re.IGNORECASE,
)

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
    q_words = set(re.findall(r"[a-z0-9]+", q))
    for term in glossary.values():
        label = _normalize(str(term.get("label", "")))
        if not label:
            continue
        label_words = re.findall(r"[a-z0-9]+", label)
        if len(label_words) == 1 and label_words[0] in q_words:
            return True
        if len(label_words) > 1 and label in q:
            return True
    words = set(re.findall(r"[a-z0-9]+", q))
    return bool(words & KG_HINTS)


def _looks_like_kg_query(question: str, matched_text: str = "") -> bool:
    q = _normalize(question)
    text = _normalize(matched_text or question)
    words = set(re.findall(r"[a-z0-9]+", q))
    text_words = set(re.findall(r"[a-z0-9]+", text))
    if words & QUERY_TASK_HINTS:
        return True
    if text_words & KG_HINTS and len(text_words) > 1:
        return True
    if re.search(r"\b(by|for|in|per|across|between|grouped by|broken down by)\b", q):
        return True
    if re.search(r"\bhow\s+(many|much)\b", q):
        return True
    return False


def _unsupported_relative_time_route(question: str, glossary: Dict[str, Dict[str, str]]) -> Optional[Dict[str, object]]:
    if not any(pattern.search(question or "") for pattern in UNSUPPORTED_RELATIVE_TIME_PATTERNS):
        return None
    if not _contains_domain_term(question, glossary):
        return None
    return {
        "route": "controlled_no_answer",
        "answer": (
            "I cannot answer this exact request because the graph does not define a rolling "
            "relative time window such as the past three months. The available time dimensions "
            "are explicit graph periods such as months, quarters, years, and time-period labels. "
            "Please ask with a concrete month, quarter, year, or supported breakdown."
        ),
        "confidence": "High",
        "reason": (
            "Unsupported relative-time window. The True Demand KG contains explicit time-period "
            "values, but not a live/current 'past N months' window."
        ),
    }


def _unsupported_live_time_route(question: str, glossary: Dict[str, Dict[str, str]]) -> Optional[Dict[str, object]]:
    q = _normalize(question)
    if not any(pattern.search(q) for pattern in UNSUPPORTED_LIVE_TIME_PATTERNS):
        return None
    if "current demand" in q or "current-demand" in q:
        return None
    if not _contains_domain_term(question, glossary):
        return None
    return {
        "route": "controlled_no_answer",
        "answer": (
            "I cannot answer live, latest, or real-time requests from the current graph. "
            "The True Demand KG contains explicit survey and time-period records, not a "
            "live operational feed. Please ask for a concrete month, quarter, year, or "
            "a supported graph breakdown."
        ),
        "confidence": "High",
        "reason": (
            "Unsupported live/latest time request. The graph is a fixed RDF dataset with "
            "explicit time-period values, not a real-time data source."
        ),
    }


def _vague_trend_route(question: str, glossary: Dict[str, Dict[str, str]]) -> Optional[Dict[str, object]]:
    q = _normalize(question)
    if not TREND_INTENT_PATTERN.search(q):
        return None
    if not _contains_domain_term(question, glossary):
        return None
    if SUPPORTED_TIME_OR_BREAKDOWN_PATTERN.search(q):
        return None
    if any(pattern.search(q) for pattern in UNSUPPORTED_RELATIVE_TIME_PATTERNS):
        return None
    options: List[Dict[str, str]] = [
        {
            "id": "quarter",
            "label": "Show semiconductor demand by quarter.",
            "rewritten_question": "Show semiconductor demand by quarter.",
        },
        {
            "id": "region_quarter",
            "label": "Show semiconductor demand percentage change by region and quarter.",
            "rewritten_question": "Show summed percentage change in semiconductor demand for each region per quarter.",
        },
        {
            "id": "future_tech_quarter",
            "label": "Show future semiconductor demand by technology category and quarter.",
            "rewritten_question": "Show future semiconductor demand by technology category and quarter.",
        },
    ]
    return {
        "route": "clarification_needed",
        "answer": "",
        "request_clarification": {
            "needs_clarification": True,
            "reason": (
                "The question asks for a trend, but it does not specify a supported graph "
                "breakdown such as quarter, region, technology category, or vehicle type."
            ),
            "question": "Which trend view should be used?",
            "options": options,
        },
        "confidence": "Low",
    }


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
            if _looks_like_kg_query(question, match.group(1)):
                return None
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
        if _looks_like_kg_query(question, match.group(1)):
            return None
        term = match.group(1).strip(" .?!")
        return {
            "route": "general_definition",
            "term": term,
            "confidence": "Medium",
            "reason": "The question asks for a general definition of a term that is not present in the Infineon KG glossary.",
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

    dr_definition = route_dr_ontology_definition(question)
    if dr_definition is not None and dr_definition.get("confidence") == "Medium":
        # Sourced via a broad keyword-scan fallback, not a precise phrase
        # match. When the question is really comparing/describing the two
        # data sources themselves (mentions both "True Demand" and "Digital
        # Reference"), let the rest of this routing chain (which classifies
        # this as a graph/source-scope question) handle it instead of a
        # generic "here are two separate definitions" dump.
        q_lower = question.lower()
        mentions_true_demand_src = bool(re.search(r"\b(true demand|demand graph|kg|knowledge graph|graph data)\b", q_lower))
        mentions_dr_src = bool(re.search(r"\b(digital reference|dr ontology)\b", q_lower))
        if mentions_true_demand_src and mentions_dr_src:
            dr_definition = None
    if dr_definition is not None:
        return dr_definition

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

    unsupported_time = _unsupported_relative_time_route(question, glossary)
    if unsupported_time is not None:
        return unsupported_time

    unsupported_live_time = _unsupported_live_time_route(question, glossary)
    if unsupported_live_time is not None:
        return unsupported_live_time

    vague_trend = _vague_trend_route(question, glossary)
    if vague_trend is not None:
        return vague_trend

    unknown_definition = _unknown_definition_route(question)
    if unknown_definition is not None:
        return unknown_definition

    underspecified = _underspecified_route(question, glossary)
    if underspecified is not None:
        return underspecified

    return {"route": "kg_query"}
