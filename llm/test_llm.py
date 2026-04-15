#test_llm.py
from dotenv import load_dotenv
load_dotenv(".env")
from rdflib import Graph
from kg.schema import load_schema
from llm.candidate_generation import generate_candidates
import json

g = Graph()
g.parse("data/infineon/graph.ttl", format="turtle")
print(f"Graph loaded: {len(g)} triples")

schema = load_schema("data/infineon/schema.json")

with open("data/infineon/infineon_dataset_30.json", "r") as f:
    dataset = json.load(f)

item = dataset[0]
question = item["question"]
gold_query = item["query"]

PREFIX = """
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

full_gold = PREFIX + gold_query if "PREFIX" not in gold_query else gold_query
gold_rows = list(g.query(full_gold))

print(f"\n=== QUESTION ===")
print(question)
print(f"\n=== GOLD RESULTS ({len(gold_rows)} rows) ===")
for row in gold_rows[:5]:
    print(f"  {tuple(row)}")

print("\nCalling LLM... please wait...")
result = generate_candidates(question, schema, k=3)
print(f"Candidates: {len(result['candidates'])}")

print(f"\n=== CANDIDATES ===")
for i, c in enumerate(result["candidates"]):
    query = c["query"]
    full_query = PREFIX + query if "PREFIX" not in query else query
    print(f"\n--- Candidate {i+1} ---")
    print(query)
    try:
        rows = list(g.query(full_query))
        match = "✅ MATCH!" if rows == gold_rows else "❌ No match"
        print(f"Results: {len(rows)} rows {match}")
        for row in rows[:3]:
            print(f"  {tuple(row)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")