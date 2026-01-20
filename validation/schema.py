from typing import Dict, List

from kg.constraints import validate_cypher
from kg.schema import KGSchema


def validate_query_schema(
    query: str, schema: KGSchema
) -> List[Dict[str, str]]:
    errors = validate_cypher(query, schema)
    return [{"type": "schema", "message": err} for err in errors]
