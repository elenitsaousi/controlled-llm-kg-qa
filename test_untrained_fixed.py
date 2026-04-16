# test_untrained_fixed.py
from dotenv import load_dotenv
load_dotenv(".env")
import os
os.environ.setdefault("LLM_PROVIDER", "infineon")

import json
import numpy as np
import joblib
from rdflib import Graph
from llm.candidate_generation import generate_candidates
from kg.schema import load_schema
from ranking.feature_extraction import extract_features
from ranking.feature_config import FEATURE_NAMES

# Load Infineon schema
schema = load_schema("data/infineon/schema.json")
with open("data/infineon/schema.json") as f:
    schema_dict = json.load(f)

# Load Infineon graph
print("Loading graph...")
g = Graph()
g.parse("data/infineon/graph.ttl", format="turtle")
print(f"Graph loaded: {len(g)} triples")

# Load ML ranker
ranker_data = joblib.load('ranking/models/infineon_ranker.joblib')
model = ranker_data['model']
scaler = ranker_data['scaler']

# SPARQL prefix
PREFIX = (
    "PREFIX survey: <http://www.semanticweb.org/gibajajulena/"
    "ontologies/2025/9/OEM_Monthly_Survey/>\n"
    "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
)

# Load test questions
with open('untrained_test.json') as f:
    tests = json.load(f)

for test in tests:
    print(f"\n{'='*50}")
    print(f"ID: {test['id']}")
    print(f"Q: {test['question']}")
    print(f"{'='*50}")

    # Step 1: Generate candidates
    try:
        cands = generate_candidates(test['question'], schema, k=3)
        candidates = cands.get('candidates', [])
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        continue

    if not candidates:
        print("⚠️ No candidates generated!")
        continue

    print(f"Generated {len(candidates)} candidates")

    # Step 2: Extract features + ML ranking
    features_list = []
    for cand in candidates:
        try:
            feats = extract_features(
                test['question'], cand['query'], schema_dict
            )
        except Exception:
            feats = {name: 0.0 for name in FEATURE_NAMES}
        features_list.append(feats)

    X_test = np.array([
        [feats.get(name, 0.0) for name in FEATURE_NAMES]
        for feats in features_list
    ])
    X_test_scaled = scaler.transform(X_test)
    scores = model.predict_proba(X_test_scaled)[:, 1]

    # Step 3: Select top candidate
    top_idx = int(np.argmax(scores))
    top_query = candidates[top_idx]['query']
    top_score = scores[top_idx]

    print(f"\n✅ Selected Candidate {top_idx+1} (score: {top_score:.3f})")
    print(f"Query: {top_query[:200]}...")

    # Step 4: Execute SPARQL
    full_query = PREFIX + top_query if "PREFIX" not in top_query else top_query
    try:
        rows = list(g.query(full_query))
        print(f"\n📊 Results: {len(rows)} rows")
        for row in rows[:5]:
            print(f"  {tuple(str(v) for v in row)}")
        if len(rows) > 5:
            print(f"  ... and {len(rows)-5} more")
    except Exception as e:
        print(f"❌ Execution error: {e}")


    # Step 5: NL Answer Synthesis
    print(f"\n💬 Answer:")
    if rows:
        # Χρησιμοποίησε το LLM για NL απάντηση
        from llm.client import InfineonGPTClient
        client = InfineonGPTClient()
        
        # Φτιάξε το synthesis prompt
        results_text = "\n".join([str(tuple(str(v) for v in row)) for row in rows[:10]])
        synthesis_prompt = f"""Given the question: "{test['question']}"
    And the SPARQL query results:
    {results_text}

    Provide a concise natural language answer in 1-2 sentences.
    Do not mention SPARQL or technical details.
    Just answer the question directly based on the data."""

        try:
            nl_answer = client.generate(synthesis_prompt, k=1)
            if nl_answer:
                print(f"  {nl_answer[0]}")
        except Exception as e:
            # Fallback: simple template answer
            print(f"  Based on the data: {results_text}")
        
        # Explainability
        print(f"\n🔍 Explanation:")
        print(f"  Selected query {top_idx+1} with confidence score {top_score:.3f}")
        print(f"  Query uses: {top_query[:150]}...")
    else:
        print("  No results found for this question.")


    # Step 5: Show all candidates with scores
    print(f"\n📋 All candidates ranked:")
    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )
    for rank, (score, cand) in enumerate(ranked):
        print(f"  {rank+1}. Score={score:.3f}: {cand['query'][:100]}...")

    print('---')