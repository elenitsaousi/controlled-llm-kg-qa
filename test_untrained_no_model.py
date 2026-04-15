import json
from llm.candidate_generation import generate_candidates
from kg.schema import load_default_schema
from kg.executor import execute_query_stub

from dotenv import load_dotenv
import os

load_dotenv()

schema = load_default_schema()

with open('untrained_test.json') as f:
    tests = json.load(f)

for test in tests:
    print(f"\n=== {test['id']} =====")
    print(f"Q: {test['question']}")
    
    # Generate candidates
    cands = generate_candidates(test['question'], schema, k=3)
    
    print("Generated candidates:")
    for i, cand in enumerate(cands['candidates']):
        print(f"  {i}: {cand['query']}")
    
    # Pick first (no ML for now)
    top_query = cands['candidates'][0]['query']
    print(f"\nTop query (no ML):")
    print(top_query)
    print()
    
    # Execute
    results = execute_query_stub(top_query, question=test['question'])
    print(f"Results: {len(results['rows'])} rows")
    if results['rows']:
        print(f"Sample: {results['rows'][0]}")
    print(f"Matched QID: {results['matched_question_id']}")
    print('---')
