# ranking/feature_extraction.py
import json
import os
import re
import contextlib
import io
import numpy as np
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Set
from ranking.feature_config import FEATURE_NAMES

LABEL_KEYWORDS = {
    "Supplier": ["supplier", "suppliers"],
    "Material": ["material", "materials"],
    "Product": ["product", "products"],
    "Yield": ["yield", "yields"],
    "Lot": ["lot", "lots"],
    "Tool": ["tool", "tools"],
    "Defect": ["defect", "defects", "defective"],
    "ProcessStep": ["process step", "process steps", "process", "step", "steps"],
    "Fab": ["fab", "fabs", "factory", "factories", "plant", "plants"],
    "CapacityConstraint": [
        "capacity constraint",
        "capacity constraints",
        "capacity",
        "constraint",
        "constraints",
        "bottleneck",
        "bottlenecks",
    ],
    "Shipment": ["shipment", "shipments"],
    "Order": ["order", "orders"],
    "Status": ["status", "statuses", "delayed", "pending", "unresolved"],
    "Inventory": ["inventory", "inventories", "stock", "stocks"],
    # Infineon-specific classes / entities
    "Region": ["region", "regions", "regional", "geographic"],
    "DemandForRegion": ["regional demand", "demand by region", "demand per region", "region demand"],
    "Quarter": ["quarter", "quarters", "q1", "q2", "q3", "q4"],
    "Company": ["company", "companies", "firms"],
    "TechnologyCategory": ["technology category", "technology node", "node size", "nm node"],
    "OrderCancellation": ["order cancellation", "order cancellations", "cancellation", "cancel"],
    "AutonomousDrivingDevelopment": ["autonomous driving", "adas", "sae level"],
    "AutonomousDrivingDevelopment_OEM": ["oem autonomous", "oem autonomous driving"],
    "AutonomousDrivingDevelopment_Tier1": ["tier1 autonomous", "tier 1 autonomous driving"],
    "InventoryDevelopment_Tier1": ["tier1 inventory", "tier 1 inventory", "inventory trend"],
    "FutureDemandAnalysis": ["future demand", "demand forecast", "demand projection"],
    "CurrentDemandAnalysis": ["current demand", "current demand change"],
    "OEM_Survey": ["oem"],
    "Tier1_Survey": ["tier1", "tier 1"],
    "Semiconductor_Survey": ["semiconductor", "semi survey"],
}

RELATION_KEYWORDS = {
    "SUPPLIES": ["supply", "supplies", "provide", "provides", "provided"],
    "AFFECTS": ["affect", "affects", "impact", "impacts", "influence", "influences"],
    "REQUIRES": ["require", "requires", "required", "need", "needs"],
    "USED_IN": ["used in", "used for", "use in", "use for", "used to"],
    "PRODUCED_FOR": ["produced for", "made for"],
    "PROCESSED_WITH": ["processed with", "process with"],
    "HAS_DEFECT": ["defect", "defects", "defective"],
    "PRODUCES": ["produce", "produces", "manufacture", "manufactures", "manufactured"],
    "HAS_CONSTRAINT": ["constraint", "constraints", "capacity", "bottleneck"],
    "CONTAINS": ["contain", "contains", "include", "includes", "included"],
    "DEPENDS_ON": ["depend on", "depends on", "dependency", "dependent on"],
    "HAS_STATUS": ["status", "delayed", "pending", "unresolved"],
    "STOCKS": ["stock", "stocks", "inventory", "inventories"],
    # Infineon-specific predicates
    "hasSurveyOrigin": ["survey origin", "survey", "oem", "tier1", "semiconductor"],
    "inRegion": ["in region", "region", "regional"],
    "regionName": ["region name"],
    "totalDemand": ["total demand", "demand volume"],
    "totalDemandPercentageChange": ["demand percentage change", "percentage change", "demand trend"],
    "percentageChange": ["percentage change", "change", "demand trend"],
    "forTimePeriod": ["time period", "quarter", "year"],
    "periodLabel": ["period label", "quarter label"],
    "reportsShortage": ["shortage", "shortages"],
    "forTechnologyCategory": ["technology category", "technology node", "node"],
    "participantCount": ["participant count", "response count", "responses"],
    "baselineType": ["baseline", "bl1", "bl2", "option1", "option2", "option3"],
    "hasVehicleType": ["vehicle type", "bev", "ice", "behv"],
    "hasSAELevel": ["sae", "sae level"],
    "hasYear": ["year"],
    "forComponent": ["component", "ev component", "non-ev component"],
    "inventoryTrend": ["inventory trend", "inventory"],
}

AMBIGUOUS_STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "type",
    "with",
}

GENERIC_SCHEMA_TOKENS = {
    "analysis",
    "assigned",
    "baseline",
    "change",
    "current",
    "data",
    "demand",
    "development",
    "future",
    "has",
    "indicator",
    "response",
    "status",
    "survey",
    "total",
    "trend",
    "type",
    "value",
}

DOMAIN_HINT_TOKENS = {
    "adas",
    "autonomous",
    "behv",
    "bev",
    "cancellation",
    "company",
    "component",
    "ice",
    "inventory",
    "nm",
    "oem",
    "order",
    "quarter",
    "region",
    "sae",
    "semiconductor",
    "technology",
    "tier1",
    "vehicle",
}

# Infineon-specific named instances [7]
INFINEON_NAMED_INSTANCES = [
    "Tier1CurrentDemand",
    "OEMCurrentDemand",
    "SemiCurrentDemand",
    "SemiFutureDemand_Option1",
    "SemiFutureDemand_Option2",
    "SemiFutureDemand_Option3",
    "Tier1FutureDemand_Option1",
    "Tier1FutureDemand_Option2",
    "Tier1FutureDemand_Option3",
    "OEMFutureDemand_Option1",
    "OEMFutureDemand_Option2",
    "OEMFutureDemand_Option3",
]

# Infineon-specific survey origins [7]
INFINEON_SURVEY_ORIGINS = [
    "OEM_Survey",
    "Tier1_Survey",
    "Semiconductor_Survey",
]

VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
PREF_NAME_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_-]*:)?[A-Za-z_][A-Za-z0-9_%-]*$"
)
FILTER_PATTERN = re.compile(r"\bFILTER\b", re.IGNORECASE)
FILTER_REMOVE = re.compile(r"\bFILTER\s*\(.*?\)", re.IGNORECASE | re.DOTALL)
OPTIONAL_REMOVE = re.compile(r"\bOPTIONAL\b", re.IGNORECASE)
SELECT_PATTERN = re.compile(r"\bSELECT\b(.*?)\bWHERE\b", re.IGNORECASE | re.DOTALL)
GROUP_BY_PATTERN = re.compile(r"\bGROUP\s+BY\b(.*?)(?:\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|$)", re.IGNORECASE | re.DOTALL)
ORDER_BY_PATTERN = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
LIMIT_PATTERN = re.compile(r"\bLIMIT\s+\d+\b", re.IGNORECASE)
AGG_PATTERN = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(", re.IGNORECASE)
SURVEY_QNAME_RE = re.compile(r"survey:([A-Za-z_][A-Za-z0-9_%/-]*)")
CAMEL_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|_|$)|[A-Z]?[a-z]+|[0-9]+")

# Lazy load sentence transformer to avoid slow startup
_semantic_model = None
_semantic_model_failed = False
_semantic_embedding_cache: Dict[str, np.ndarray] = {}
_semantic_similarity_cache: Dict[str, float] = {}

def _get_semantic_model():
    """Lazy load sentence transformer model."""
    global _semantic_model, _semantic_model_failed
    if os.getenv("ENABLE_SEMANTIC_EMBEDDING", "0") != "1":
        return None
    if _semantic_model_failed:
        return None
    if _semantic_model is None:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                from sentence_transformers import SentenceTransformer

                _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _semantic_model = None
            _semantic_model_failed = True
    return _semantic_model


def _embedding_cache_key(text: str) -> str:
    return normalize_text(text)[:4096]


def _encode_for_similarity(text: str) -> Optional[np.ndarray]:
    model = _get_semantic_model()
    if model is None:
        return None
    key = _embedding_cache_key(text)
    cached = _semantic_embedding_cache.get(key)
    if cached is not None:
        return cached
    emb = np.asarray(model.encode([key])[0], dtype=float)
    norm = float(np.linalg.norm(emb))
    if norm <= 1e-12:
        return None
    emb = emb / norm
    if len(_semantic_embedding_cache) < 10000:
        _semantic_embedding_cache[key] = emb
    return emb


def _query_text_for_embedding(query: str) -> str:
    """Turn SPARQL into a compact semantic summary before embedding it."""
    q = re.sub(r"(?im)^\s*PREFIX\s+.*$", " ", str(query or ""))
    q = re.sub(r"https?://[^\s>]+[/#]", " ", q)
    q = q.replace("?", " ")
    tokens: List[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_%-]*", q):
        if raw.upper() in {
            "SELECT",
            "WHERE",
            "PREFIX",
            "ORDER",
            "BY",
            "GROUP",
            "LIMIT",
            "FILTER",
            "BIND",
            "AS",
            "OPTIONAL",
            "DISTINCT",
        }:
            continue
        if raw.lower() in {"xsd", "rdf", "rdfs", "survey", "http", "www"}:
            continue
        expanded = re.sub(r"[_%-]+", " ", raw)
        expanded = re.sub(r"(?<!^)(?=[A-Z])", " ", expanded)
        tokens.extend(t.lower() for t in expanded.split() if len(t) > 1)

    aggregations = [m.lower() for m in AGG_PATTERN.findall(str(query or ""))]
    if "GROUP BY" in str(query or "").upper():
        aggregations.append("group by")
    if "ORDER BY" in str(query or "").upper():
        aggregations.append("ranking")
    return " ".join(aggregations + tokens)


def load_schema(schema_path: str) -> Dict[str, object]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def keyword_in_text(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(r"\b" + re.escape(keyword) + r"\b", text) is not None


def _split_camel_tokens(value: str) -> List[str]:
    norm = value.replace("_", " ").replace("-", " ").replace("%28", " ").replace("%29", " ")
    out: List[str] = []
    for part in norm.split():
        out.extend(CAMEL_TOKEN_RE.findall(part))
    return [t.lower() for t in out if t]


def _keywords_from_schema_term(term: str) -> List[str]:
    tokens = [t for t in _split_camel_tokens(term) if t not in AMBIGUOUS_STOP_WORDS]
    if not tokens:
        return []
    keywords: List[str] = []
    full = " ".join(tokens)
    if len(full) >= 4:
        keywords.append(full)
    if len(tokens) >= 2:
        compact = " ".join(t for t in tokens if t not in {"for", "to"})
        if compact and compact != full:
            keywords.append(compact)
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) >= 3 and token not in GENERIC_SCHEMA_TOKENS:
            keywords.append(token)
    else:
        for token in tokens:
            if token in DOMAIN_HINT_TOKENS:
                keywords.append(token)
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for kw in keywords:
        if kw in seen:
            continue
        seen.add(kw)
        unique.append(kw)
    return unique


def _normalize_qname(token: str) -> str:
    t = token.strip().strip("()[]{}.,;")
    if not t or t.startswith("?"):
        return ""
    if t.startswith("<") and t.endswith(">"):
        core = t[1:-1]
        if "#" in core:
            return core.rsplit("#", 1)[-1]
        if "/" in core:
            return core.rsplit("/", 1)[-1]
        return core
    if ":" in t:
        return t.split(":", 1)[1]
    return t


def _is_pref_name(token: str) -> bool:
    t = token.strip().strip("()[]{}.,;")
    return bool(PREF_NAME_RE.match(t))


def _split_outside(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    in_single = False
    in_double = False
    escape = False

    for ch in text:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            continue

        if not in_single and not in_double:
            if ch == "(":
                depth_paren += 1
            elif ch == ")" and depth_paren > 0:
                depth_paren -= 1
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]" and depth_bracket > 0:
                depth_bracket -= 1
            elif ch == "{":
                depth_brace += 1
            elif ch == "}" and depth_brace > 0:
                depth_brace -= 1

            if (
                ch == sep
                and depth_paren == 0
                and depth_bracket == 0
                and depth_brace == 0
            ):
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def extract_question_entities(question: str, schema_labels: Iterable[str]) -> Set[str]:
    q = normalize_text(question)
    entities = set()
    for label in schema_labels:
        keywords = LABEL_KEYWORDS.get(label)
        if keywords is None:
            keywords = _keywords_from_schema_term(label)
        for kw in keywords:
            if keyword_in_text(q, kw):
                entities.add(label)
                break
    return entities


def extract_question_relations(
    question: str,
    schema_predicates: Optional[Sequence[str]] = None,
) -> Set[str]:
    q = normalize_text(question)
    relations = set()
    for rel, keywords in RELATION_KEYWORDS.items():
        for kw in keywords:
            if keyword_in_text(q, kw):
                relations.add(rel)
                break
    if schema_predicates:
        for pred in schema_predicates:
            pred_keywords = _keywords_from_schema_term(pred)
            for kw in pred_keywords:
                if keyword_in_text(q, kw):
                    relations.add(pred)
                    break
    return relations


def extract_query_labels(query: str) -> Set[str]:
    labels = set()
    for _, pred, obj in extract_triples(query):
        if pred.lower() in {"a", "rdf:type"}:
            normalized = _normalize_qname(obj)
            if normalized:
                labels.add(normalized)
    return labels


def extract_query_relations(query: str) -> Set[str]:
    rels = set()
    for _, pred, _ in extract_triples(query):
        pred_lower = pred.lower()
        if pred_lower in {"a", "rdf:type"}:
            continue
        normalized = _normalize_qname(pred)
        if normalized:
            rels.add(normalized)
    return rels


def extract_select_vars(query: str) -> Set[str]:
    match = SELECT_PATTERN.search(query)
    if match:
        return set(VAR_PATTERN.findall(match.group(1)))
    return set(VAR_PATTERN.findall(query))


def extract_group_by_vars(query: str) -> Set[str]:
    match = GROUP_BY_PATTERN.search(query)
    if not match:
        return set()
    return set(VAR_PATTERN.findall(match.group(1)))


def extract_aggregations(query: str) -> Set[str]:
    return {m.group(1).upper() for m in AGG_PATTERN.finditer(query or "")}


def extract_survey_qnames(query: str) -> Set[str]:
    return {_normalize_qname(m.group(1)) for m in SURVEY_QNAME_RE.finditer(query or "")}


def extract_query_plan(query: str, schema: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Extract a compact, supervised-learning-friendly query plan from SPARQL."""
    schema_classes = set((schema or {}).get("classes", []))
    schema_terms = extract_survey_qnames(query)
    triple_classes = extract_query_labels(query)
    predicates = extract_query_relations(query)
    triples = extract_triples(query)
    aggregations = extract_aggregations(query)
    group_by_vars = extract_group_by_vars(query)
    select_vars = extract_select_vars(query)
    query_upper = query.upper()

    survey_origins = {
        s
        for s in schema_terms | triple_classes
        if s in {"Tier1_Survey", "OEM_Survey", "Semiconductor_Survey"}
    }
    classes = set(triple_classes)
    if schema_classes:
        classes.update(t for t in schema_terms if t in schema_classes)

    group_by_predicates: Set[str] = set()
    for subj, pred, obj in triples:
        if obj in group_by_vars:
            if pred.lower() in {"a", "rdf:type"}:
                continue
            pred_name = _normalize_qname(pred)
            if pred_name:
                group_by_predicates.add(pred_name)

    query_types = set()
    if aggregations:
        query_types.add("aggregation")
    if group_by_vars:
        query_types.add("grouped")
    if "FILTER" in query_upper or "VALUES" in query_upper:
        query_types.add("filtered")
    if "UNION" in query_upper:
        query_types.add("union")
    if ORDER_BY_PATTERN.search(query):
        query_types.add("ordered")
    if LIMIT_PATTERN.search(query):
        query_types.add("limited")
    if any(a in aggregations for a in {"MIN", "MAX"}) or LIMIT_PATTERN.search(query):
        query_types.add("ranking")

    labels = set()
    labels.update(f"class:{x}" for x in sorted(classes))
    labels.update(f"predicate:{x}" for x in sorted(predicates))
    labels.update(f"survey:{x}" for x in sorted(survey_origins))
    labels.update(f"aggregation:{x}" for x in sorted(aggregations))
    labels.update(f"group_by_var:{x.lstrip('?')}" for x in sorted(group_by_vars))
    labels.update(f"group_by_predicate:{x}" for x in sorted(group_by_predicates))
    labels.update(f"query_type:{x}" for x in sorted(query_types))

    return {
        "classes": sorted(classes),
        "predicates": sorted(predicates),
        "survey_origins": sorted(survey_origins),
        "aggregations": sorted(aggregations),
        "group_by_vars": sorted(v.lstrip("?") for v in group_by_vars),
        "group_by_predicates": sorted(group_by_predicates),
        "select_vars": sorted(v.lstrip("?") for v in select_vars),
        "query_types": sorted(query_types),
        "labels": sorted(labels),
    }


def extract_triples(query: str) -> List[tuple]:
    if "{" in query and "}" in query:
        body = query.split("{", 1)[1].rsplit("}", 1)[0]
    else:
        body = query
    body = FILTER_REMOVE.sub(" ", body)
    body = OPTIONAL_REMOVE.sub(" ", body)
    body = body.replace("{", " ").replace("}", " ")
    statements = _split_outside(body, ".")
    triples = []
    for stmt in statements:
        if not stmt:
            continue
        segments = _split_outside(stmt, ";")
        if not segments:
            continue

        head_parts = segments[0].split()
        if len(head_parts) < 3:
            continue
        head_token_upper = head_parts[0].upper()
        if head_token_upper.startswith(("BIND", "FILTER", "VALUES", "OPTIONAL", "UNION")):
            continue

        subj = head_parts[0]
        pred = head_parts[1]
        pred_norm = pred.lower()
        if not (
            pred_norm in {"a", "rdf:type"}
            or pred.startswith("?")
            or _is_pref_name(pred)
        ):
            continue
        obj_expr = " ".join(head_parts[2:])
        for obj in _split_outside(obj_expr, ","):
            obj_val = obj.strip()
            if obj_val:
                triples.append((subj, pred, obj_val))

        for seg in segments[1:]:
            seg_parts = seg.split()
            if len(seg_parts) < 2:
                continue
            pred = seg_parts[0]
            pred_norm = pred.lower()
            if not (
                pred_norm in {"a", "rdf:type"}
                or pred.startswith("?")
                or _is_pref_name(pred)
            ):
                continue
            obj_expr = " ".join(seg_parts[1:])
            for obj in _split_outside(obj_expr, ","):
                obj_val = obj.strip()
                if obj_val:
                    triples.append((subj, pred, obj_val))
    return triples


def build_undirected_schema_graph(schema: Dict[str, object]) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for rel in schema.get("relationships", []):
        for src in rel.get("from", []):
            for dst in rel.get("to", []):
                graph[src].add(dst)
                graph[dst].add(src)
    return graph


def shortest_path(
    graph: Dict[str, Set[str]], start: str, goal: str
) -> List[str]:
    if start == goal:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        for neighbor in graph.get(node, set()):
            if neighbor == goal:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return []


def expected_intermediates(
    question_entities: Set[str], schema: Dict[str, object]
) -> Set[str]:
    graph = build_undirected_schema_graph(schema)
    entities = list(question_entities)
    intermediates: Set[str] = set()
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            path = shortest_path(graph, entities[i], entities[j])
            if len(path) > 2:
                intermediates.update(path[1:-1])
    return intermediates


def _semantic_similarity(question: str, query: str) -> float:
    """
    Semantic similarity between the question and a readable query summary. [7]
    Uses sentence transformers if enabled and available; otherwise returns 0.0.
    """
    try:
        candidate_text = _query_text_for_embedding(query)
        if not candidate_text:
            return 0.0
        cache_key = f"{_embedding_cache_key(question)}\n---\n{_embedding_cache_key(candidate_text)}"
        cached = _semantic_similarity_cache.get(cache_key)
        if cached is not None:
            return cached
        q_emb = _encode_for_similarity(question)
        c_emb = _encode_for_similarity(candidate_text)
        if q_emb is None or c_emb is None:
            return 0.0
        score = float(np.dot(q_emb, c_emb))
        if len(_semantic_similarity_cache) < 50000:
            _semantic_similarity_cache[cache_key] = score
        return score
    except Exception:
        return 0.0


def extract_features(question: str, query: str, schema: Dict[str, object]) -> Dict[str, float]:
    schema_labels = list(schema.get("classes") or (schema.get("labels") or {}).keys())
    schema_predicates = list(
        set(schema.get("predicates", [])) | set(schema.get("properties", []))
    )

    q_entities_raw = extract_question_entities(question, schema_labels)
    q_relations_raw = extract_question_relations(question, schema_predicates=schema_predicates)

    q_entities = { _normalize_qname(x) for x in q_entities_raw if _normalize_qname(x) }
    q_relations = { _normalize_qname(x) for x in q_relations_raw if _normalize_qname(x) }

    qry_labels = extract_query_labels(query)
    qry_relations = extract_query_relations(query)
    triples = extract_triples(query)

    # --- GOLD-AWARE FEATURES ---
    from kg.sparql_matching import extract_requirements
    gold_query = schema.get("current_gold_query", "")
    if gold_query:
        reqs = extract_requirements(gold_query)
        gold_classes = {_normalize_qname(x) for x in reqs["classes"] if _normalize_qname(x)}
        gold_preds = {_normalize_qname(x) for x in reqs["predicates"] if _normalize_qname(x)}
        cand_classes = qry_labels
        cand_preds = qry_relations
        predicate_overlap = len(cand_preds & gold_preds)
        class_overlap = len(cand_classes & gold_classes)
        missing_predicates = len(gold_preds - cand_preds)
        extra_predicates = len(cand_preds - gold_preds)
    else:
        predicate_overlap = 0
        class_overlap = 0
        missing_predicates = 0
        extra_predicates = 0

    select_vars = extract_select_vars(query)
    allowed_predicates = {
        _normalize_qname(x)
        for x in schema_predicates
        if _normalize_qname(x)
    }
    intermediates = expected_intermediates(q_entities, schema)
    intermediates_norm = {
        _normalize_qname(x) for x in intermediates if _normalize_qname(x)
    }
    variables = VAR_PATTERN.findall(query)
    node_count = len(set(variables))
    rel_count = sum(1 for _, pred, _ in triples if pred.lower() not in {"a", "rdf:type"})
    triple_count = len(triples)
    type_triple_count = sum(
        1 for _, pred, _ in triples if pred.lower() in {"a", "rdf:type"}
    )
    has_type = int(type_triple_count > 0)
    distinct_label_count = len(qry_labels)
    distinct_relation_count = len(qry_relations)
    query_upper = query.upper()
    has_variable_length = int("*" in query_upper)
    has_optional = int("OPTIONAL" in query_upper)
    has_where = int("WHERE" in query_upper)
    has_exists = int("EXISTS" in query_upper)
    has_distinct = int("DISTINCT" in query_upper)
    has_aggregation = int(
        any(k in query_upper for k in [
            "COUNT(", "SUM(", "AVG(", "MIN(", "MAX(", "GROUP_CONCAT("
        ])
    )
    property_filter_count = len(FILTER_PATTERN.findall(query))

    vars_in_triples = set()
    graph: Dict[str, Set[str]] = {}
    invalid_predicate_count = 0

    for subj, pred, obj in triples:
        if subj.startswith("?"):
            vars_in_triples.add(subj)
            graph.setdefault(subj, set())
        if obj.startswith("?"):
            vars_in_triples.add(obj)
            graph.setdefault(obj, set())
        if subj.startswith("?") and obj.startswith("?"):
            graph[subj].add(obj)
            graph[obj].add(subj)
        pred_lower = pred.lower()
        if pred_lower in {"a", "rdf:type"}:
            continue
        pred_name = _normalize_qname(pred)
        if pred_name and pred_name not in allowed_predicates:
            invalid_predicate_count += 1

    component_count = 0
    visited = set()
    for var in vars_in_triples:
        if var in visited:
            continue
        component_count += 1
        stack = [var]
        visited.add(var)
        while stack:
            node = stack.pop()
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

    has_disconnected = int(component_count > 1)
    unused_select_vars = len([v for v in select_vars if v not in vars_in_triples])
    entity_overlap = len(q_entities & qry_labels)
    relation_overlap = len(q_relations & qry_relations)
    entity_coverage = entity_overlap / len(q_entities) if q_entities else 0.0
    entity_precision = entity_overlap / len(qry_labels) if qry_labels else 0.0
    relation_coverage = relation_overlap / len(q_relations) if q_relations else 0.0
    relation_precision = relation_overlap / len(qry_relations) if qry_relations else 0.0
    expected_intermediate_coverage = (
        len(intermediates_norm & qry_labels) / len(intermediates_norm)
        if intermediates_norm
        else 0.0
    )
    unexpected_labels = qry_labels - q_entities - intermediates_norm
    unexpected_label_ratio = (
        len(unexpected_labels) / len(qry_labels) if qry_labels else 0.0
    )

    # --- Infineon-specific features [7] ---
    has_named_instance = int(
        any(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(inst)}(?![A-Za-z0-9_])",
                query,
            )
            for inst in INFINEON_NAMED_INSTANCES
        )
    )
    uses_correct_survey = int(
        any(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(s)}(?![A-Za-z0-9_])",
                query,
            )
            for s in INFINEON_SURVEY_ORIGINS
        )
    )
    semantic_sim = _semantic_similarity(question, query)

    features = {
        "node_count": float(node_count),
        "rel_count": float(rel_count),
        "triple_count": float(triple_count),
        "type_triple_count": float(type_triple_count),
        "has_type": float(has_type),
        "component_count": float(component_count),
        "has_disconnected": float(has_disconnected),
        "invalid_predicate_count": float(invalid_predicate_count),
        "unused_select_vars": float(unused_select_vars),
        "distinct_label_count": float(distinct_label_count),
        "distinct_relation_count": float(distinct_relation_count),
        "has_variable_length": float(has_variable_length),
        "has_optional": float(has_optional),
        "has_where": float(has_where),
        "has_exists": float(has_exists),
        "has_distinct": float(has_distinct),
        "has_aggregation": float(has_aggregation),
        "property_filter_count": float(property_filter_count),
        "entity_coverage": float(entity_coverage),
        "entity_precision": float(entity_precision),
        "relation_coverage": float(relation_coverage),
        "relation_precision": float(relation_precision),
        "expected_intermediate_coverage": float(expected_intermediate_coverage),
        "unexpected_label_ratio": float(unexpected_label_ratio),
        "predicate_overlap": float(predicate_overlap),
        "class_overlap": float(class_overlap),
        "missing_predicates": float(missing_predicates),
        "extra_predicates": float(extra_predicates),
        # Infineon-specific [7]
        "has_named_instance": float(has_named_instance),
        "uses_correct_survey": float(uses_correct_survey),
        "semantic_similarity": float(semantic_sim),
    }

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return features
