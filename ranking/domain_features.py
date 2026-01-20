# ranking/domain_features.py

"""
Domain-aware feature extraction for candidate queries.
This module implements lightweight, explainable heuristics
based on domain semantics (supply chain / manufacturing).
"""

def extract_domain_features(query: str, question_type: str) -> dict:
    q = query.upper()

    features = {}

    features["expected_path"] = has_expected_path(q, question_type)
    features["forbidden_shortcut"] = has_forbidden_shortcut(q, question_type)
    features["relevant_relations"] = has_relevant_relations(q)
    features["simple_structure"] = is_simple_structure(q)
    features["entity_constraint"] = has_entity_constraint(q)


    return features


# ---------- Feature definitions ----------

def has_expected_path(q: str, question_type: str) -> int:
    if question_type == "SUPPLIER_WAFER_PRODUCT":
        return int(
            "SUPPLIER" in q
            and ("WAFER" in q or "MATERIAL" in q)
            and "PRODUCT" in q
        )
    if question_type == "FACTORY_PRODUCT":
        return int("FACTORY" in q or "FAB" in q and "PRODUCT" in q)
    return 0



def has_forbidden_shortcut(q: str, question_type: str) -> int:
    if question_type == "SUPPLIER_WAFER_PRODUCT":
        return int(
            "SUPPLIER" in q
            and "PRODUCT" in q
            and not ("WAFER" in q or "MATERIAL" in q)
        )
    return 0

def has_entity_constraint(q: str) -> int:
    return int("{" in q and "}" in q)



RELEVANT_RELATIONS = [
    "SUPPLIES",
    "USED_IN",
    "MANUFACTURES",
    "PRODUCES"
]

FORBIDDEN_RELATIONS = [
    "HAS_DEFECT",
    "HAS_YIELD",
    "AFFECTS"
]

def has_relevant_relations(q: str) -> int:
    if any(rel in q for rel in FORBIDDEN_RELATIONS):
        return 0
    return int(any(rel in q for rel in RELEVANT_RELATIONS))


COMPLEX_ENTITIES = [
    "LOT",
    "DEFECT",
    "TOOL",
    "YIELD"
]

def is_simple_structure(q: str) -> int:
    return int(not any(ent in q for ent in COMPLEX_ENTITIES))
