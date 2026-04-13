from dotenv import load_dotenv
load_dotenv()

from rdflib import Graph
from kg.schema import load_default_schema
from llm.candidate_generation import generate_candidates

# Load graph
g = Graph()
g.parse("data/infineon/graph.ttl", format="turtle")
print(f"Triples loaded: {len(g)}")

# Load schema (temporary toy)
schema = load_default_schema()

# Generate candidates
res = generate_candidates(
    "Which companies are OEM?",
    schema,
    k=3
)

# Run candidates
PREFIX = """
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""
correct = 0
valid_wrong = 0
invalid = 0

for i, c in enumerate(res["candidates"]):
    query = c["query"]

    query = query.replace(":", "survey:")
    query = PREFIX + query

    print(f"\n--- Candidate {i+1} ---")
    print(query)

    try:
        results = g.query(query)
        rows = list(results)

        if rows:
            label = "correct"
            correct += 1
            print(f"✅ Results: {len(rows)}")
        else:
            label = "valid_wrong"
            valid_wrong += 1
            print("⚠️ No results")

    except Exception as e:
        label = "invalid"
        invalid += 1
        print(f"❌ ERROR: {e}")

    print(f"Label: {label}")

print("\n--- RAW OUTPUT ---")
print("\n--- SUMMARY ---")
print(f"Correct: {correct}")
print(f"Valid but wrong: {valid_wrong}")
print(f"Invalid: {invalid}")
print(res)