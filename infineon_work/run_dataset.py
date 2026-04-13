from dotenv import load_dotenv
load_dotenv()

import json
from rdflib import Graph
from llm.candidate_generation import generate_candidates
from kg.schema import load_default_schema

# --- Load KG ---
g = Graph()
g.parse("data/infineon/graph.ttl", format="turtle")
print(f"Triples loaded: {len(g)}")

# --- Load dataset ---
with open("data/infineon/infineon_dataset_30.json") as f:
    dataset = json.load(f)

# --- Schema (temporary) ---
schema = load_default_schema()

# --- PREFIX ---
PREFIX = """
PREFIX : <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def _ensure_prefixes(query: str) -> str:
    if "PREFIX" in query.upper():
        return query
    return PREFIX + query

# --- counters ---
correct = 0
valid_wrong = 0
invalid = 0
total = 0

# --- loop ---
for item in dataset:
    question = item["question"]
    print("\n==============================")
    print(f"Question: {question}")

    res = generate_candidates(question, schema, k=3)

    for c in res["candidates"]:
        query = c["query"]

        query = _ensure_prefixes(query)

        try:
            results = g.query(query)
            rows = list(results)

            if rows:
                label = "correct"
                correct += 1
            else:
                label = "valid_wrong"
                valid_wrong += 1

        except Exception:
            label = "invalid"
            invalid += 1

        total += 1
        print(f"Label: {label}")

# --- summary ---
print("\n===== FINAL RESULTS =====")
print(f"Total queries: {total}")
print(f"Correct: {correct}")
print(f"Valid but wrong: {valid_wrong}")
print(f"Invalid: {invalid}")

print("\nPercentages:")
print(f"Correct: {correct/total:.2%}")
print(f"Valid wrong: {valid_wrong/total:.2%}")
print(f"Invalid: {invalid/total:.2%}")
