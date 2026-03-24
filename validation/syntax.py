import re
from typing import Dict, List

_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_WHERE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)

_SELECT_KEYWORDS = {
    "SELECT",
    "DISTINCT",
    "REDUCED",
    "AS",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "SAMPLE",
    "GROUP_CONCAT",
}


def _balanced(text: str, left: str, right: str) -> bool:
    return text.count(left) == text.count(right)


def _select_clause(query: str) -> str:
    match = re.search(r"\bSELECT\b([\s\S]*?)\bWHERE\b", query, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def _has_invalid_select_vars(select_clause: str) -> bool:
    if not select_clause:
        return False
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\?\w+|\*", select_clause)
    for tok in tokens:
        if tok == "*":
            continue
        if tok.upper() in _SELECT_KEYWORDS:
            continue
        if tok.startswith("?"):
            continue
        return True
    return False


def validate_query_syntax(query: str) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    if not _SELECT_RE.search(query):
        errors.append(
            {"type": "syntax", "message": "Missing SELECT clause."}
        )
    if not _WHERE_RE.search(query):
        errors.append(
            {"type": "syntax", "message": "Missing WHERE clause."}
        )
    select_clause = _select_clause(query)
    if select_clause and _has_invalid_select_vars(select_clause):
        errors.append(
            {
                "type": "syntax",
                "message": "All SELECT variables must start with '?'.",
            }
        )
    if not _balanced(query, "(", ")"):
        errors.append(
            {"type": "syntax", "message": "Unbalanced parentheses."}
        )
    if not _balanced(query, "{", "}"):
        errors.append(
            {"type": "syntax", "message": "Unbalanced braces."}
        )
    if not _balanced(query, "[", "]"):
        errors.append(
            {"type": "syntax", "message": "Unbalanced brackets."}
        )
    return errors
