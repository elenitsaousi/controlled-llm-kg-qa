import re
from typing import Dict, List

_MATCH_RE = re.compile(r"\bMATCH\b", re.IGNORECASE)
_RETURN_RE = re.compile(r"\bRETURN\b", re.IGNORECASE)


def _balanced(text: str, left: str, right: str) -> bool:
    return text.count(left) == text.count(right)


def validate_query_syntax(query: str) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not _MATCH_RE.search(query):
        errors.append(
            {"type": "syntax", "message": "Missing MATCH clause."}
        )
    if not _RETURN_RE.search(query):
        errors.append(
            {"type": "syntax", "message": "Missing RETURN clause."}
        )
    if not _balanced(query, "(", ")"):
        errors.append(
            {"type": "syntax", "message": "Unbalanced parentheses."}
        )
    if not _balanced(query, "{", "}"):
        errors.append(
            {"type": "syntax", "message": "Unbalanced property braces."}
        )
    return errors
