from rdflib import Graph

# Load graph
g = Graph()
g.parse("data/infineon/graph.ttl", format="turtle")

print(f"Triples loaded: {len(g)}")

# Simple test query
query = """
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>

SELECT ?company
WHERE {
    ?company a survey:Company .
}
LIMIT 10
"""

results = g.query(query)

for row in results:
    print(row)