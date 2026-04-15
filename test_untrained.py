import json
from llm.candidate_generation import generate_candidates
from kg.schema import load_default_schema
from kg.executor import execute_query_stub
import xgboost as xgb
import numpy as np
from ranking.feature_extraction import extract_features
from ranking.feature_config import FEATURE_NAMES

schema = load_default_schema()
model = load_model('ranking/ml_learning_ranker/train_xgb.npy')

with open('untrained_test.json') as f:
    tests = json.load(f)

for test in tests:
    print(f"\n=== {test['id']} =====")
    print(f"Q: {test['question']}")
    
    # Generate candidates
    cands = generate_candidates(test['question'], schema, k=3)
    
    # Extract features, rank
    features = []
    for cand in cands['candidates']:
        feats = extract_features(test['question'], cand['query'], schema)  # from ranking.feature_extraction
        features.append(feats)
    
    scores = model.predict(features)
    
    # Top candidate
    top_idx = np.argmax(scores)
    top_query = cands['candidates'][top_idx]['query']
    
    print(f"Top ranked query: {top_query}")
    
    # Execute
    results = execute_sparql(top_query, 'data/infineon/graph.ttl')
    print(f"Results: {len(results)} rows")
    print(f"First result: {list(results[0].values()) if results else 'EMPTY'}")

