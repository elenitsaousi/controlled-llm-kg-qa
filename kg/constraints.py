import re
from typing import List

from kg.schema import KGSchema

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
