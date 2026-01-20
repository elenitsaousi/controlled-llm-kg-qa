from typing import Dict, List

from kg.schema import KGSchema
from validation.schema import validate_query_schema
from validation.semantic import validate_query_semantic
from validation.syntax import validate_query_syntax


def extract_features(query: str, schema: KGSchema) -> Dict[str, float]:
    syntax_errors = validate_query_syntax(query)
    schema_errors = validate_query_schema(query, schema)
    semantic_errors = validate_query_semantic(query)
    error_count = len(syntax_errors) + len(schema_errors) + len(semantic_errors)
    return {
        "error_count": float(error_count),
        "length": float(len(query)),
    }


def collect_errors(query: str, schema: KGSchema) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    errors.extend(validate_query_syntax(query))
    errors.extend(validate_query_schema(query, schema))
    errors.extend(validate_query_semantic(query))
    return errors


def score_candidate(query: str, schema: KGSchema) -> float:
    features = extract_features(query, schema)
    penalty = (10.0 * features["error_count"]) + (0.001 * features["length"])
    return -penalty
