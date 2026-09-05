from typing import Dict, List

from kg.constraints import validate_sparql
from kg.schema import KGSchema


def validate_query_schema(
    query: str, schema: KGSchema
) -> List[Dict[str, str]]:
    errors = validate_sparql(query, schema)
    return [{"type": "schema", "message": err} for err in errors]
