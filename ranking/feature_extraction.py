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

NODE_PATTERN = re.compile(r"\(([^)]*)\)")
LABEL_PATTERN = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")
REL_PATTERN = re.compile(r"\[:([A-Za-z_][A-Za-z0-9_|]*)")
REL_BLOCK_PATTERN = re.compile(r"\[[^\]]*\]")
PROP_PATTERN = re.compile(r"\{[^}]+\}")


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
    labels = set()
    for node in NODE_PATTERN.findall(query):
        for label in LABEL_PATTERN.findall(node):
            labels.add(label)
    return labels


def extract_query_relations(query: str) -> Set[str]:
    rels = set()
    for rel_group in REL_PATTERN.findall(query):
        for rel in rel_group.split("|"):
            if rel:
                rels.add(rel)
    return rels


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


def extract_features(question: str, query: str, schema: Dict[str, object]) -> Dict[str, float]:
    schema_labels = schema.get("labels", {}).keys()

    q_entities = extract_question_entities(question, schema_labels)
    q_relations = extract_question_relations(question)
    qry_labels = extract_query_labels(query)
    qry_relations = extract_query_relations(query)
    intermediates = expected_intermediates(q_entities, schema)

    node_count = len(NODE_PATTERN.findall(query))
    rel_count = len(REL_BLOCK_PATTERN.findall(query))
    distinct_label_count = len(qry_labels)
    distinct_relation_count = len(qry_relations)

    query_upper = query.upper()
    has_variable_length = int(any("*" in block for block in REL_BLOCK_PATTERN.findall(query)))
    has_optional = int("OPTIONAL MATCH" in query_upper)
    has_where = int("WHERE" in query_upper)
    has_exists = int("EXISTS" in query_upper)
    has_distinct = int("DISTINCT" in query_upper)
    has_aggregation = int(
        any(k in query_upper for k in ["COUNT(", "COLLECT(", "SUM(", "AVG(", "MIN(", "MAX("])
    )
    property_filter_count = len(PROP_PATTERN.findall(query))

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

    features = {
        "node_count": float(node_count),
        "rel_count": float(rel_count),
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
    }

    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    return features
