from kg.schema import KGSchema
from ranking.feature_extraction import extract_question_entities, extract_question_relations


def build_candidate_prompt(question: str, schema: KGSchema, k: int = 5) -> str:
    schema_text = schema.as_prompt_text()
    schema_labels = schema.classes or schema.labels.keys()
    detected_entities = sorted(extract_question_entities(question, schema_labels))
    detected_relations = sorted(extract_question_relations(question))
    hints = []
    if detected_entities:
        hints.append("Detected entities: " + ", ".join(detected_entities))
    if detected_relations:
        hints.append("Detected relations: " + ", ".join(detected_relations))
    hints_text = "\n".join(hints)

    return (
        "You generate SPARQL SELECT queries for an RDF knowledge graph.\n\n"

        "STRICT REQUIREMENTS:\n"
        "- You MUST follow STRICT SPARQL syntax\n"
        "- Each triple MUST end with a dot '.'\n"
        "- All predicates MUST be prefixed with ':' (e.g., :SUPPLIES)\n"
        "- You MUST include rdf:type using 'a' for ALL main entities\n"
        "- Do NOT omit rdf:type\n"
        "- All variables MUST start with '?'\n"
        "- Use WHERE { ... } block\n"
        "- Use ONLY the classes, predicates, and properties from the schema\n"
        "- Do NOT invent new classes or predicates\n"
        "- Use single quotes in FILTER clauses\n\n"

        "DIVERSITY REQUIREMENTS:\n"
        "- Generate structurally diverse candidates (different joins, predicates, or graph structure)\n"
        "- Include a mix of correct, partially correct, and incorrect queries\n"
        "- Do NOT output near-duplicates or minor rephrasings\n\n"

        "STYLE REQUIREMENTS:\n"
        "- NEVER place a literal directly in a triple; always bind a variable and use FILTER\n"
        "- Candidate 1 MUST be the most likely correct and minimal query\n"
        "- Candidate 1 MUST include all detected entities and relations (if provided)\n"
        "- Avoid OPTIONAL unless explicitly needed\n\n"

        f"{hints_text}\n\n"

        "OUTPUT FORMAT:\n"
        f"- Return EXACTLY {k} queries\n"
        "- Output MUST be a valid JSON array of strings\n"
        "- Do NOT include explanations\n\n"

        "CORRECT EXAMPLE:\n"
        "Question: Which suppliers supply products?\n"
        "Output:\n"
        "[\n"
        "  \"SELECT ?supplier WHERE { ?supplier a :Supplier . ?supplier :SUPPLIES ?product . ?product a :Product . }\",\n"
        "  \"SELECT ?supplier WHERE { ?supplier a :Supplier . ?supplier :PRODUCES ?product . ?product a :Product . }\",\n"
        "  \"SELECT ?supplier WHERE { ?supplier a :Supplier . ?supplier :SUPPLIES ?product . ?product a :Material . }\",\n"
        "  \"SELECT ?supplier WHERE { ?supplier a :Supplier . ?supplier :SUPPLIES ?product . ?product a :Product . ?product :REQUIRES ?tool . ?tool a :Tool . }\",\n"
        "  \"SELECT ?supplier WHERE { ?supplier a :Supplier . ?supplier :DEPENDS_ON ?supplier2 . ?supplier2 a :Supplier . }\"\n"
        "]\n\n"

        "INCORRECT EXAMPLES (DO NOT DO THIS):\n"
        "- SELECT ?s WHERE { ?s SUPPLIES ?p }\n"
        "- Missing ':' before predicates\n"
        "- Missing '.' at end of triples\n"
        "- Missing rdf:type\n\n"
        "- ?x :name 'X' (use FILTER instead)\n\n"

        f"{schema_text}\n\n"

        f"Question: {question}\n"
        "Output:\n"
    )
