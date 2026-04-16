# run_query.py
from dotenv import load_dotenv
load_dotenv(".env")
from rdflib import Graph

# Load graph
g = Graph()
g.parse("data/infineon/graph.ttl", format="turtle")
print(f"Graph loaded: {len(g)} triples")

PREFIX = """
PREFIX survey: <http://www.semanticweb.org/gibajajulena/ontologies/2025/9/OEM_Monthly_Survey/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

# Βάλε εδώ το query που θέλεις να τεστάρεις
# Tier1 current demand BL1/BL2
query = """
SELECT ?baseline ?pct WHERE {
  survey:Tier1CurrentDemand
    survey:hasAggregatedResult ?entry .
  ?entry survey:baselineType ?baseline ;
         survey:percentageChange ?pct .
}
"""

full_query = PREFIX + query
try:
    rows = list(g.query(full_query))
    print(f"\nResults: {len(rows)} rows")
    for row in rows:
        print(f"  {tuple(str(v) for v in row)}")
except Exception as e:
    print(f"Error: {e}")