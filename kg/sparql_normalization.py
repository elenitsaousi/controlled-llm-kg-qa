import itertools
import re
from typing import Dict, Iterable, List, Tuple


VAR_PATTERN = re.compile(r"\?[A-Za-z_][A-Za-z0-9_]*")
FILTER_RE = re.compile(r"FILTER\s*\((.*?)\)", re.IGNORECASE | re.DOTALL)


def _strip_prefix_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().upper().startswith("PREFIX"):
            continue
        lines.append(line)
    return " ".join(lines)


def _collapse_ws(text: str) -> str:
    return " ".join(text.strip().split())


def _split_optional_blocks(where_body: str) -> Tuple[str, List[str]]:
    remaining_parts: List[str] = []
    optional_blocks: List[str] = []

    i = 0
    n = len(where_body)
    while i < n:
        match = re.search(r"\bOPTIONAL\b", where_body[i:], re.IGNORECASE)
        if not match:
            remaining_parts.append(where_body[i:])
            break

        start = i + match.start()
        remaining_parts.append(where_body[i:start])

        brace_start = where_body.find("{", start)
        if brace_start == -1:
            remaining_parts.append(where_body[start:])
            break

        depth = 0
        j = brace_start
        while j < n:
            if where_body[j] == "{":
                depth += 1
            elif where_body[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1

        if depth != 0:
            remaining_parts.append(where_body[start:])
            break

        optional_blocks.append(where_body[brace_start + 1 : j])
        i = j + 1

    return "".join(remaining_parts), optional_blocks


def _extract_filters(block: str) -> Tuple[str, List[str]]:
    filters: List[str] = []

    def _capture(match: re.Match) -> str:
        expr = match.group(1)
        expr = re.sub(r"\s+", " ", expr.strip())
        expr = re.sub(r"\s*([=!<>]+)\s*", r"\1", expr)
        filters.append(expr)
        return " "

    cleaned = FILTER_RE.sub(_capture, block)
    return cleaned, filters


def _parse_triples(block: str) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for stmt in block.split("."):
        stmt = stmt.strip()
        if not stmt:
            continue
        parts = stmt.split()
        if len(parts) < 3:
            continue
        subj, pred, obj = parts[0], parts[1], parts[2]
        if pred.lower() == "rdf:type":
            pred = "a"
        triples.append((subj, pred, obj))
    return triples


def _apply_var_mapping(text: str, mapping: Dict[str, str]) -> str:
    def _replace(match: re.Match) -> str:
        var = match.group(0)
        return mapping.get(var, var)

    return VAR_PATTERN.sub(_replace, text)


def _canonical_representation(
    select_vars: List[str],
    triples: List[Tuple[bool, str, str, str]],
    filters: List[Tuple[bool, str]],
    distinct: bool,
) -> str:
    variables = sorted({v for v in VAR_PATTERN.findall(" ".join(select_vars))})
    if not variables:
        variables = sorted(
            {
                v
                for t in triples
                for v in VAR_PATTERN.findall(" ".join(t[1:]))
            }
        )
    if not variables:
        select_part = ",".join(sorted(select_vars)) if select_vars else ""
        triple_repr = "|".join(
            sorted(
                [
                    f"{'O' if opt else 'R'}:{s} {p} {o}"
                    for opt, s, p, o in triples
                ]
            )
        )
        filter_repr = "|".join(
            sorted(
                [f"{'O' if opt else 'R'}:{expr}" for opt, expr in filters]
            )
        )
        return (
            f"DISTINCT:{int(distinct)};"
            f"SELECT:{select_part};"
            f"TRIPLES:{triple_repr};"
            f"FILTERS:{filter_repr}"
        )

    canonical_names = [f"?v{i+1}" for i in range(len(variables))]
    best: str = ""
    first = True
    for perm in itertools.permutations(variables):
        mapping = {var: canonical_names[i] for i, var in enumerate(perm)}

        mapped_select = [_apply_var_mapping(v, mapping) for v in select_vars]
        mapped_select_sorted = ",".join(sorted(mapped_select))

        triple_repr = []
        for opt, s, p, o in triples:
            s_m = _apply_var_mapping(s, mapping)
            o_m = _apply_var_mapping(o, mapping)
            triple_repr.append(f"{'O' if opt else 'R'}:{s_m} {p} {o_m}")
        triple_repr = "|".join(sorted(triple_repr))

        filter_repr = []
        for opt, expr in filters:
            expr_m = _apply_var_mapping(expr, mapping)
            filter_repr.append(f"{'O' if opt else 'R'}:{expr_m}")
        filter_repr = "|".join(sorted(filter_repr))

        rep = (
            f"DISTINCT:{int(distinct)};"
            f"SELECT:{mapped_select_sorted};"
            f"TRIPLES:{triple_repr};"
            f"FILTERS:{filter_repr}"
        )
        if first or rep < best:
            best = rep
            first = False

    return best


def normalize_sparql(query: str) -> str:
    if not query:
        return ""

    text = _strip_prefix_lines(query)
    text = _collapse_ws(text)

    upper = text.upper()
    select_idx = upper.find("SELECT")
    if select_idx == -1:
        return text.lower()

    distinct = "SELECT DISTINCT" in upper

    where_idx = upper.find("WHERE", select_idx)
    brace_idx = text.find("{", where_idx if where_idx != -1 else select_idx)

    select_part = text[select_idx + len("SELECT") : where_idx if where_idx != -1 else brace_idx]
    select_part = select_part.strip()
    select_vars = VAR_PATTERN.findall(select_part)
    if "*" in select_part and "*" not in select_vars:
        select_vars = ["*"]

    if brace_idx == -1:
        return _collapse_ws(text)

    brace_end = text.rfind("}")
    if brace_end == -1 or brace_end <= brace_idx:
        return _collapse_ws(text)

    where_body = text[brace_idx + 1 : brace_end]
    where_body = _collapse_ws(where_body)

    mandatory_body, optional_blocks = _split_optional_blocks(where_body)

    triples: List[Tuple[bool, str, str, str]] = []
    filters: List[Tuple[bool, str]] = []

    def parse_block(block: str, optional: bool) -> None:
        cleaned, block_filters = _extract_filters(block)
        for expr in block_filters:
            filters.append((optional, expr))
        for s, p, o in _parse_triples(cleaned):
            triples.append((optional, s, p, o))

    parse_block(mandatory_body, False)
    for block in optional_blocks:
        parse_block(block, True)

    return _canonical_representation(select_vars, triples, filters, distinct)

