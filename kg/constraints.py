import re
from typing import List

from kg.schema import KGSchema

_TYPE_RE = re.compile(r"\b(?:a|rdf:type)\s+(:[A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_PRED_RE = re.compile(
    r"\?[A-Za-z_][A-Za-z0-9_]*\s+(:[A-Za-z_][A-Za-z0-9_]*)\s+",
    re.IGNORECASE,
)


def _strip_prefix(term: str) -> str:
    return term[1:] if term.startswith(":") else term


def validate_sparql(query: str, schema: KGSchema) -> List[str]:
    errors: List[str] = []

    # Classes used with rdf:type / a
    classes = [_strip_prefix(t) for t in _TYPE_RE.findall(query)]
    for cls in classes:
        if not schema.class_allowed(cls):
            errors.append(f"Unknown class: {cls}")

    # Predicates used in triple patterns
    preds = [_strip_prefix(p) for p in _PRED_RE.findall(query)]
    for pred in preds:
        if not schema.predicate_allowed(pred):
            errors.append(f"Unknown predicate: {pred}")

    return errors


# Backwards compatibility (Cypher)
_NODE_LABEL_RE = re.compile(r"\([A-Za-z0-9_]*:([A-Za-z0-9_]+)")
_REL_TYPE_RE = re.compile(r"\[:([A-Za-z0-9_]+)\]")
_LABEL_PROPS_RE = re.compile(r":([A-Za-z0-9_]+)\s*\{([^}]*)\}")


def _extract_prop_keys(props_text: str) -> List[str]:
    keys: List[str] = []
    for chunk in props_text.split(","):
        if ":" not in chunk:
            continue
        key = chunk.split(":", 1)[0].strip()
        if key:
            keys.append(key)
    return keys


def validate_cypher(query: str, schema: KGSchema) -> List[str]:
    errors: List[str] = []

    labels = _NODE_LABEL_RE.findall(query)
    for label in labels:
        if not schema.label_allowed(label):
            errors.append(f"Unknown label: {label}")

    rel_types = _REL_TYPE_RE.findall(query)
    for rel_type in rel_types:
        if not schema.relationship_allowed(rel_type):
            errors.append(f"Unknown relationship type: {rel_type}")

    for label, props_text in _LABEL_PROPS_RE.findall(query):
        for prop in _extract_prop_keys(props_text):
            if not schema.property_allowed(label, prop):
                errors.append(f"Property '{prop}' not allowed on label '{label}'")

    return errors
