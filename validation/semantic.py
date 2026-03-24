import re
from typing import Dict, List

_WRITE_RE = re.compile(r"\b(INSERT|DELETE|UPDATE)\b", re.IGNORECASE)


def validate_query_semantic(query: str) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if _WRITE_RE.search(query):
        errors.append(
            {
                "type": "semantic",
                "message": "Write operations are not allowed in read-only QA.",
            }
        )
    return errors
