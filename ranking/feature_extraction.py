# ranking/feature_extraction.py
import json
import re
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set
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
TYPE_PATTERN = re.compile(
    r"\b(?:a|rdf:type)\s+(:[A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
PRED_PATTERN = re.compile(
    r"\?[A-Za-z_][A-Za-z0-9_]*\s+(:[A-Za-z_][A-Za-z0-9_]*)\s+",
    re.IGNORECASE,
)
FILTER_PATTERN = re.compile(r"\bFILTER\b", re.IGNORECASE)
FILTER_REMOVE = re.compile(r"\bFILTER\s*\(.*?\)", re.IGNORECASE | re.DOTALL)
OPTIONAL_REMOVE = re.compile(r"\bOPTIONAL\b", re.IGNORECASE)
SELECT_PATTERN = re.compile(r"\bSELECT\b(.*?)\bWHERE\b", re.IGNORECASE | re.DOTALL)

# Lazy load sentence transformer to avoid slow startup
_semantic_model = None

def _get_semantic_model():
    """Lazy load sentence transformer model."""
    global _semantic_model
    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _semantic_model = None
    return _semantic_model


def load_schema(schema_path: str) -> Dict[str, object]:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def keyword_in_text(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(r"\b" + re.escape(keyword) + r"\b", text) is not None


def extract_question_entities(question: str, schema_labels: Iterable[str]) -> Set[str]:
    q = normalize_text(question)
    entities = set()
    for label in schema_labels:
        keywords = LABEL_KEYWORDS.get(label, [label.lower()])
        for kw in keywords:
            if keyword_in_text(q, kw):
                entities.add(label)
                break
    return entities


def extract_question_relations(question: str) -> Set[str]:
    q = normalize_text(question)
    relations = set()
    for rel, keywords in RELATION_KEYWORDS.items():
        for kw in keywords:
            if keyword_in_text(q, kw):
                relations.add(rel)
                break
    return relations


def extract_query_labels(query: str) -> Set[str]:
    return {match[1:] for match in TYPE_PATTERN.findall(query)}


def extract_query_relations(query: str) -> Set[str]:
    return {match[1:] for match in PRED_PATTERN.findall(query)}


def extract_select_vars(query: str) -> Set[str]:
    match = SELECT_PATTERN.search(query)
    if match:
        return set(VAR_PATTERN.findall(match.group(1)))
    return set(VAR_PATTERN.findall(query))


def extract_triples(query: str) -> List[tuple]:
    if "{" in query and "}" in query:
        body = query.split("{", 1)[1].rsplit("}", 1)[0]
    else:
        body = query
    body = FILTER_REMOVE.sub(" ", body)
    body = OPTIONAL_REMOVE.sub(" ", body)
    statements = [s.strip() for s in body.split(".") if s.strip()]
    triples = []
    for stmt in statements:
        parts = stmt.split()
        if len(parts) < 3:
            continue
        triples.append((parts[0], parts[1], parts[2]))
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
    Semantic similarity between question and SPARQL query. [7]
    Uses sentence transformers if available, fallback to 0.0
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        model = _get_semantic_model()
        if model is None:
            return 0.0
        q_emb = model.encode([question])
        c_emb = model.encode([query])
        return float(cosine_similarity(q_emb, c_emb)[0][0])
    except Exception:
        return 0.0


def extract_features(question: str, query: str, schema: Dict[str, object]) -> Dict[str, float]:
    schema_labels = schema.get("classes") or schema.get("labels", {}).keys()
    q_entities = extract_question_entities(question, schema_labels)
    q_relations = extract_question_relations(question)
    qry_labels = extract_query_labels(query)
    qry_relations = extract_query_relations(query)
    triples = extract_triples(query)

    # --- GOLD-AWARE FEATURES ---
    from kg.sparql_matching import extract_requirements
    gold_query = schema.get("current_gold_query", "")
    if gold_query:
        reqs = extract_requirements(gold_query)
        gold_classes = set(reqs["classes"])
        gold_preds = set(reqs["predicates"])
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
    allowed_predicates = set(schema.get("predicates", [])) | set(schema.get("properties", []))
    intermediates = expected_intermediates(q_entities, schema)
    variables = VAR_PATTERN.findall(query)
    node_count = len(set(variables))
    rel_count = len(PRED_PATTERN.findall(query))
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
        pred_name = pred[1:] if pred.startswith(":") else pred
        if pred_name not in allowed_predicates:
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
        len(intermediates & qry_labels) / len(intermediates) if intermediates else 0.0
    )
    unexpected_labels = qry_labels - q_entities - intermediates
    unexpected_label_ratio = (
        len(unexpected_labels) / len(qry_labels) if qry_labels else 0.0
    )

    # --- Infineon-specific features [7] ---
    has_named_instance = int(
        any(inst in query for inst in INFINEON_NAMED_INSTANCES)
    )
    uses_correct_survey = int(
        any(s in query for s in INFINEON_SURVEY_ORIGINS)
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