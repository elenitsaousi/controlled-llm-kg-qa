from kg.schema import KGSchema


def build_candidate_prompt(question: str, schema: KGSchema, k: int = 5) -> str:
    schema_text = schema.as_prompt_text()
    return (
        "You generate Cypher queries for a knowledge graph.\n"
        "Use ONLY the labels, relationships, and properties listed in the schema.\n"
        "Do not invent labels, relationship types, or properties.\n"
        "Use single quotes for string literals.\n"
        f"Return exactly {k} candidate queries as a JSON array of strings.\n\n"
        f"{schema_text}\n\n"
        f"Question: {question}\n"
        "Candidates:"
    )
