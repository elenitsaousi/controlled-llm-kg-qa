import json
from pathlib import Path
from typing import Dict, List, Tuple


BASE = Path(__file__).resolve().parents[3]
QUESTIONS_PATH = BASE / "data" / "toy_kg" / "questions" / "questions.json"
SCHEMA_PATH = BASE / "data" / "toy_kg" / "schema.json"
OUT_DIR = BASE / "data" / "toy_kg" / "experiments" / "sparql_candidates"


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_sparql(query: str, properties: List[str]) -> dict:
    select_part = query.split("WHERE", 1)[0]
    where_part = query.split("{", 1)[1].rsplit("}", 1)[0]

    select_tokens = select_part.replace("SELECT", "").replace("DISTINCT", "")
    select_vars = [t.strip() for t in select_tokens.split() if t.strip()]

    statements = [s.strip() for s in where_part.split(".") if s.strip()]
    types: Dict[str, str] = {}
    rel_triples: List[Tuple[str, str, str]] = []
    prop_triples: List[Tuple[str, str, str]] = []
    filters: Dict[str, str] = {}

    for stmt in statements:
        if stmt.upper().startswith("FILTER"):
            # FILTER(?var = 'X')
            inner = stmt[stmt.find("(") + 1 : stmt.rfind(")")]
            if "=" in inner:
                var, value = inner.split("=", 1)
                var = var.strip()
                value = value.strip().strip("'").strip('"')
                filters[var] = value
            continue
        parts = stmt.split()
        if len(parts) < 3:
            continue
        subj, pred, obj = parts[0], parts[1], parts[2]
        if pred == "a":
            types[subj] = obj.lstrip(":")
        else:
            pred_name = pred.lstrip(":")
            if pred_name in properties:
                prop_triples.append((subj, pred_name, obj))
            else:
                rel_triples.append((subj, pred_name, obj))

    return {
        "select_vars": select_vars,
        "types": types,
        "rel_triples": rel_triples,
        "prop_triples": prop_triples,
        "filters": filters,
    }


def build_query(
    select_vars: List[str],
    types: Dict[str, str],
    rel_triples: List[Tuple[str, str, str]],
    prop_triples: List[Tuple[str, str, str]],
    filters: Dict[str, str],
    distinct: bool = False,
) -> str:
    parts: List[str] = []
    for var, cls in types.items():
        parts.append(f"{var} a :{cls} .")
    for s, p, o in rel_triples:
        parts.append(f"{s} :{p} {o} .")
    for s, p, o in prop_triples:
        parts.append(f"{s} :{p} {o} .")
    for var, val in filters.items():
        parts.append(f"FILTER({var} = '{val}')")

    select = "SELECT "
    if distinct:
        select += "DISTINCT "
    select += " ".join(select_vars)
    where = "WHERE { " + " ".join(parts) + " }"
    return select + " " + where


def generate_candidates(question: dict, schema: dict) -> List[str]:
    properties = schema.get("properties", [])
    gold = question.get("gold_query", "")
    distinct = "SELECT DISTINCT" in gold.upper()
    parsed = parse_sparql(gold, properties)

    candidates: List[str] = []

    # 1) Correct (gold)
    candidates.append(gold)

    # 2) Drop one relation triple (partial)
    rel_triples = list(parsed["rel_triples"])
    if rel_triples:
        dropped = rel_triples[1:] if len(rel_triples) > 1 else []
        cand = build_query(
            parsed["select_vars"],
            parsed["types"],
            dropped,
            parsed["prop_triples"],
            parsed["filters"],
            distinct=distinct,
        )
        candidates.append(cand)

    # 3) Replace one relation predicate (incorrect)
    rel_triples = list(parsed["rel_triples"])
    if rel_triples:
        all_preds = [p for p in schema.get("predicates", []) if p != rel_triples[0][1]]
        new_pred = all_preds[0] if all_preds else rel_triples[0][1]
        rel_triples[0] = (rel_triples[0][0], new_pred, rel_triples[0][2])
        cand = build_query(
            parsed["select_vars"],
            parsed["types"],
            rel_triples,
            parsed["prop_triples"],
            parsed["filters"],
            distinct=distinct,
        )
        candidates.append(cand)

    # 4) Swap relation direction (incorrect)
    rel_triples = list(parsed["rel_triples"])
    if rel_triples:
        s, p, o = rel_triples[0]
        rel_triples[0] = (o, p, s)
        cand = build_query(
            parsed["select_vars"],
            parsed["types"],
            rel_triples,
            parsed["prop_triples"],
            parsed["filters"],
            distinct=distinct,
        )
        candidates.append(cand)

    # 5) Modify filter or type (partial/incorrect)
    filters = dict(parsed["filters"])
    types = dict(parsed["types"])
    if filters:
        first_var = next(iter(filters.keys()))
        filters[first_var] = "Unknown"
    elif parsed["prop_triples"]:
        var = parsed["prop_triples"][0][2]
        filters[var] = "Unknown"
    elif types:
        # No filters or properties: swap a type to create a distinct candidate.
        var = next(iter(types.keys()))
        all_classes = [c for c in schema.get("classes", []) if c != types[var]]
        if all_classes:
            types[var] = all_classes[0]
    cand = build_query(
        parsed["select_vars"],
        types,
        parsed["rel_triples"],
        parsed["prop_triples"],
        filters,
        distinct=distinct,
    )
    candidates.append(cand)

    # Ensure 5 unique candidates
    deduped = []
    seen = set()
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
        if len(deduped) == 5:
            break

    while len(deduped) < 5:
        deduped.append(candidates[0])

    return deduped


def main() -> None:
    schema = load_schema()
    questions = load_questions()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in questions:
        qid = item["id"]
        question_text = item["question"]
        candidates = generate_candidates(item, schema)

        data = {
            "question_id": qid,
            "question": question_text,
            "candidates": [
                {"id": f"{qid}_C{i+1}", "query": q}
                for i, q in enumerate(candidates)
            ],
        }
        out_path = OUT_DIR / f"{qid}_candidates.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    print(f"Wrote {len(questions)} SPARQL candidate files to {OUT_DIR}")


if __name__ == "__main__":
    main()
