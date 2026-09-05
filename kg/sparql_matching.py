import re
from typing import Dict, List, Set, Tuple


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
FILTER_REMOVE = re.compile(r"\bFILTER\s*\(.*?\)", re.IGNORECASE | re.DOTALL)
OPTIONAL_REMOVE = re.compile(r"\bOPTIONAL\b", re.IGNORECASE)


def _strip_prefix(term: str) -> str:
    return term[1:] if term.startswith(":") else term


def _extract_body(query: str) -> str:
    if "{" in query and "}" in query:
        body = query.split("{", 1)[1].rsplit("}", 1)[0]
    else:
        body = query
    body = FILTER_REMOVE.sub(" ", body)
    body = OPTIONAL_REMOVE.sub(" ", body)
    return body


def parse_sparql(query: str) -> Tuple[Dict[str, Set[str]], List[Tuple[str, str, str]]]:
    types: Dict[str, Set[str]] = {}
    triples: List[Tuple[str, str, str]] = []

    body = _extract_body(query)
    statements = [s.strip() for s in body.split(".") if s.strip()]
    for stmt in statements:
        parts = stmt.split()
        if len(parts) < 3:
            continue
        subj, pred, obj = parts[0], parts[1], parts[2]
        pred_lower = pred.lower()
        if pred_lower in {"a", "rdf:type"}:
            if subj.startswith("?") and obj.startswith(":"):
                types.setdefault(subj, set()).add(_strip_prefix(obj))
            continue
        triples.append((subj, _strip_prefix(pred), obj))

    return types, triples


def extract_requirements(
    gold_query: str,
) -> Dict[str, object]:
    gold_types, gold_triples = parse_sparql(gold_query)
    gold_classes: Set[str] = set()
    for classes in gold_types.values():
        gold_classes.update(classes)

    required_predicates = {p for _, p, _ in gold_triples}
    required_edges: List[Tuple[str, Set[str], Set[str]]] = []
    for subj, pred, obj in gold_triples:
        subj_classes = gold_types.get(subj, set()) if subj.startswith("?") else set()
        obj_classes = gold_types.get(obj, set()) if obj.startswith("?") else set()
        required_edges.append((pred, set(subj_classes), set(obj_classes)))

    return {
        "classes": gold_classes,
        "predicates": required_predicates,
        "edges": required_edges,
    }


def _edge_matches(
    req_edge: Tuple[str, Set[str], Set[str]],
    cand_edge: Tuple[str, Set[str], Set[str], str, str],
) -> bool:
    req_pred, req_subj_classes, req_obj_classes = req_edge
    cand_pred, cand_subj_classes, cand_obj_classes, cand_subj, cand_obj = cand_edge

    if cand_pred != req_pred:
        return False

    if req_subj_classes:
        if not cand_subj_classes:
            return False
        if cand_subj_classes.isdisjoint(req_subj_classes):
            return False

    if req_obj_classes:
        if not cand_obj_classes:
            return False
        if cand_obj_classes.isdisjoint(req_obj_classes):
            return False

    return True


def is_relaxed_correct(candidate_query: str, gold_query: str) -> bool:
    if not candidate_query or not gold_query:
        return False

    reqs = extract_requirements(gold_query)
    cand_types, cand_triples = parse_sparql(candidate_query)

    cand_classes: Set[str] = set()
    for classes in cand_types.values():
        cand_classes.update(classes)

    if not reqs["classes"].issubset(cand_classes):
        return False

    cand_predicates = {p for _, p, _ in cand_triples}
    if not reqs["predicates"].issubset(cand_predicates):
        return False

    cand_edges: List[Tuple[str, Set[str], Set[str], str, str]] = []
    for subj, pred, obj in cand_triples:
        subj_classes = cand_types.get(subj, set()) if subj.startswith("?") else set()
        obj_classes = cand_types.get(obj, set()) if obj.startswith("?") else set()
        cand_edges.append((pred, set(subj_classes), set(obj_classes), subj, obj))

    for req_edge in reqs["edges"]:
        matched = any(_edge_matches(req_edge, cand_edge) for cand_edge in cand_edges)
        if not matched:
            return False

    return True
